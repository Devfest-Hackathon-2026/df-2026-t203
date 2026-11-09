"""Stage 2 of the single-video pipeline: smooth and gap-fill the landmarks.

The smoothing and gap filling are freemocap's own, not a reimplementation. This
calls `skellyforge <https://github.com/freemocap/skellyforge>`_ - the library
freemocap calls from ``skeleton_from_mediapipe_observations.py`` - in the order
freemocap uses it:

1. ``interpolate_trajectory`` - linear interpolation across every gap, per
   marker. This carries the body through frames MediaPipe lost.
2. ``filter_trajectory`` - a 4th-order Butterworth low-pass at 6 Hz, run through
   ``filtfilt`` so it has no phase lag. This is what takes the jitter out.

The sampling rate is taken from the video's real frame rate rather than left at
skellyforge's 30 Hz default - freemocap fixed that same bug in its own pipeline,
and on 60 fps footage the default would put the true cutoff at 12 Hz and pass
most of what the filter exists to remove.

Four things are done to the data before it is handed over.

**Hands are split into orientation and shape first.** MediaPipe reports hand
landmarks relative to the wrist, so turning a hand over rotates all 21 points
around the origin. Low-pass filtering those coordinates directly averages
opposite vectors and shrinks the hand towards a point; on the sample clip, a
wrist that genuinely turns 180 degrees in ten frames came out with a collapsed
hand whose orientation then swung wildly - the filter *created* a flip that was
not in the measurement. So each hand is decomposed per frame into a palm
orientation and the finger shape in palm-local coordinates. Both are well
behaved under interpolation and low-pass filtering, and the hand is rebuilt
afterwards. The body needs none of this: pose landmarks are already in a
hip-centred frame that does not spin.

**Hand gaps are filled here, not by straight-line interpolation.** A straight
line between two handshapes is not a handshape - halfway through the sample
clip's 14-frame gap the interpolated fingers had curled to 54% of their length
and twisted, none of which was measured. So while a hand is untracked its
fingers hold the last shape they were seen in (``hold_handshape_in_gaps``) and
its palm turns along the shortest arc towards where it reappears
(``slerp_orientation_gaps``).

**Frames where the palm orientation spikes are dropped.** See
``reject_orientation_spikes``.

**The clip is not trimmed**: every input frame is processed, whether or not hands are in view.

What freemocap's *solver* is not used for: freemocap reconstructs 3D by
triangulating several calibrated cameras, and its single-camera fallback
(``project_2d_batch_to_3d``) maps image pixels onto the Y=0 plane, which has no
depth at all. MediaPipe's world landmarks, which stage 1 already produces, are
the better source of 3D from one camera. So the depth estimate stays
MediaPipe's, and the smoothing comes from freemocap.

Run standalone with any Python that has skellyforge installed:

    python postprocess_landmarks.py --landmarks landmarks.json --out landmarks_filtered.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import hand_stabilizer

POSE_COUNT = 33
HAND_COUNT = 21
HAND_WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
AXIS_COUNT = 3  # the three columns of a hand's palm basis, carried as if they were markers
HAND_OFFSETS = {"Left": POSE_COUNT, "Right": POSE_COUNT + HAND_COUNT}
MARKER_COUNT = POSE_COUNT + 2 * HAND_COUNT

# Layout of the array handed to skellyforge: the pose markers, then per hand the
# 21 palm-local shape points followed by the 3 MIDbasis axes.
SOLVE_HAND_OFFSETS = {
    "Left": POSE_COUNT,
    "Right": POSE_COUNT + HAND_COUNT + AXIS_COUNT,
}
SOLVE_MARKER_COUNT = POSE_COUNT + 2 * (HAND_COUNT + AXIS_COUNT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth and gap-fill landmarks with freemocap's skellyforge post-processing.",
    )
    parser.add_argument("--landmarks", required=True, type=Path, help="landmarks.json from stage 1.")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the post-processed landmarks.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Butterworth low-pass cutoff in Hz (freemocap's default is 6.0). Lower is smoother and blunter.",
    )
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order (freemocap's default is 4).")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the sampling rate. By default the frame rate recorded in landmarks.json is used.",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=45.0,
        help=(
            "Degrees of palm rotation between frames above which a hand is treated as a bad "
            "detection, if the orientation returns to where it was. 0 disables the check."
        ),
    )
    parser.add_argument(
        "--max-spike-run",
        type=int,
        default=3,
        help="Longest run of frames the spttike check will drop in one go.",
    )
    parser.add_argument(
        "--max-wrist-speed",
        type=float,
        default=1500.0,
        help=(
            "Fastest palm rotation the avatar is allowed, in degrees per second. Jumps that "
            "demand more than this are followed over several frames instead of snapping. "
            "0 disables the limit."
        ),
    )
    parser.add_argument(
        "--handshape-in-gaps",
        choices=["hold", "interpolate"],
        default="hold",
        help=(
            "What the fingers do while a hand is not tracked. 'hold' keeps the last measured "
            "handshape; 'interpolate' morphs towards the next one, which passes through poses "
            "that were never measured."
        ),
    )
    parser.add_argument(
        "--gap-return-blend",
        type=float,
        default=0.15,
        help=(
            "Seconds at the end of a gap spent easing a held handshape into the pose the hand "
            "is next measured in, so tracking resumes without a snap. 0 holds until the last frame."
        ),
    )
    parser.add_argument(
        "--body-visibility-threshold",
        type=float,
        default=0.5,
        help=(
            "Pose landmarks (arms, legs, feet, face, ...) whose MediaPipe visibility stays below "
            "this are treated as lost and held instead of trusted. 0 disables body hold."
        ),
    )
    parser.add_argument(
        "--body-frames-to-hold",
        type=int,
        default=3,
        help="Consecutive low-visibility frames before a body landmark counts as lost.",
    )
    parser.add_argument(
        "--body-frames-to-recover",
        type=int,
        default=3,
        help="Consecutive good frames before a lost body landmark counts as tracked again.",
    )
    parser.add_argument(
        "--body-return-blend",
        type=float,
        default=0.25,
        help="Seconds at the end of a body gap spent easing the held position into the next measured one.",
    )
    parser.add_argument("--no-interpolate", action="store_true", help="Skip gap filling.")
    parser.add_argument("--no-filter", action="store_true", help="Skip the Butterworth filter.")
    parser.add_argument(
        "--confidence-gate",
        action="store_true",
        help=(
            "Drop hands that stage 2.5's confidence check rejects, before the filter runs. "
            "Worth doing whenever stage 2.5 follows: the Butterworth is filtfilt, so a bad "
            "frame left in here spreads into its neighbours and cannot be separated out later."
        ),
    )
    parser.add_argument(
        "--hand-shape-filter",
        choices=["butterworth", "none"],
        default="butterworth",
        help=(
            "'none' leaves the palm-local finger shape unfiltered for stage 2.5's One Euro "
            "pass, and filters everything else as usual. Two low-pass filters over the same "
            "signal cannot be tuned independently, so pick one."
        ),
    )
    parser.add_argument(
        "--stabilize-config",
        type=Path,
        default=None,
        help="stabilize.yaml to read the confidence thresholds from (default: next to this script).",
    )
    return parser.parse_args(argv)


def import_skellyforge():
    """Import freemocap's post-processing library, with a useful message if it is absent."""
    try:
        from skellyforge.data_models.trajectory_3d import Trajectory3d
        from skellyforge.post_processing.filters.apply_filter import filter_trajectory
        from skellyforge.post_processing.filters.filter_config import FilterConfig
        from skellyforge.post_processing.interpolation.apply_interpolation import interpolate_trajectory
        from skellyforge.post_processing.interpolation.interpolation_config import InterpolationConfig
    except ImportError as error:
        raise SystemExit(
            f"skellyforge is not importable from {sys.executable} ({error}).\n"
            "Install freemocap's post-processing library into this interpreter:\n"
            '    pip install "skellyforge @ git+https://github.com/freemocap/skellyforge@main"'
        ) from error
    return {
        "Trajectory3d": Trajectory3d,
        "filter_trajectory": filter_trajectory,
        "FilterConfig": FilterConfig,
        "interpolate_trajectory": interpolate_trajectory,
        "InterpolationConfig": InterpolationConfig,
    }


# ---------------------------------------------------------------------------
# Landmarks in and out
# ---------------------------------------------------------------------------


def build_marker_array(frames: list[dict], reliable: dict | None = None) -> np.ndarray:
    """Stack every frame's landmarks into (n_frames, 75, 3), NaN where not measured.

    `reliable` is the stage 2.5 confidence gate, ``{label: [bool per frame]}``.
    A hand it rejects is dropped here rather than later, and the reason is the
    filter further down: skellyforge's Butterworth runs through ``filtfilt``,
    which smears a bad frame both forwards and backwards in time. A garbage
    frame that survives this far contaminates its neighbours, and no amount of
    cleaning up afterwards can separate them again. Dropped hands become gaps,
    which the hold-and-slerp machinery below already knows how to cross.
    """
    data = np.full((len(frames), MARKER_COUNT, 3), np.nan, dtype=float)
    for index, frame in enumerate(frames):
        pose = frame.get("pose_world")
        if pose is not None:
            data[index, :POSE_COUNT, :] = [point[:3] for point in pose]
        for label, offset in HAND_OFFSETS.items():
            if reliable is not None and not reliable[label][index]:
                continue
            points = (frame.get("hands") or {}).get(label)
            if points is not None:
                data[index, offset:offset + HAND_COUNT, :] = points
    return data


def frames_from_array(
    data: np.ndarray, original_frames: list[dict], reliable: dict | None = None
) -> list[dict]:
    """Rebuild the per-frame JSON structure from the processed array.

    Two different honesty markers come out of here, and stage 2.5 needs both:

    ``hands_measured``  which hands MediaPipe reported on that frame.
    ``hands_reliable``  which of those survived the confidence gate. Anything in
                        the first list but not the second was detected and then
                        judged untrustworthy, so the handshape now sitting in
                        that frame was filled in, not measured.

    The pose points lose MediaPipe's visibility score, which would be
    meaningless once filtered. The per-hand confidence signals stage 1 wrote are
    carried through untouched, so re-tuning the stabiliser never means
    re-running MediaPipe.
    """
    frames = []
    for index, original in enumerate(original_frames):
        pose = data[index, :POSE_COUNT, :]
        hands = {}
        for label, offset in HAND_OFFSETS.items():
            points = data[index, offset:offset + HAND_COUNT, :]
            if np.isfinite(points).all():
                hands[label] = [[round(float(value), 5) for value in point] for point in points]
        frames.append(
            {
                "frame": original.get("frame", index),
                "pose_world": (
                    [[round(float(value), 5) for value in point] for point in pose]
                    if np.isfinite(pose).all()
                    else None
                ),
                "hands": hands,
                "hands_measured": sorted((original.get("hands") or {}).keys()),
                "hands_reliable": sorted(
                    label
                    for label in (original.get("hands") or {})
                    if reliable is None or reliable[label][index]
                ),
                "hand_confidence": original.get("hand_confidence") or {},
            }
        )
    return frames


# ---------------------------------------------------------------------------
# Palm orientation
# ---------------------------------------------------------------------------


def palm_basis(hand: np.ndarray) -> np.ndarray | None:
    """Orthonormal frame of one hand: Y along the middle finger, X across the knuckles."""
    return orthonormalise(hand[INDEX_MCP] - hand[PINKY_MCP], hand[MIDDLE_MCP] - hand[HAND_WRIST])


def orthonormalise(across: np.ndarray, along: np.ndarray) -> np.ndarray | None:
    """Right-handed basis with Y along `along` and X along the part of `across` across it."""
    if not np.isfinite(across).all() or not np.isfinite(along).all():
        return None
    length = np.linalg.norm(along)
    if length < 1e-9:
        return None
    y_axis = along / length
    x_axis = across - y_axis * np.dot(across, y_axis)
    length = np.linalg.norm(x_axis)
    if length < 1e-9:
        return None
    x_axis = x_axis / length
    return np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)], axis=1)


def rotation_between(first: np.ndarray, second: np.ndarray) -> float:
    """Angle in degrees between two orientations."""
    cosine = (np.trace(second @ first.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def reject_orientation_spikes(
    data: np.ndarray, start: int, threshold_deg: float, max_run: int
) -> int:
    """Blank frames where a hand's orientation jumps away and comes straight back.

    When the signer's two hands overlap, MediaPipe occasionally reconstructs one
    of them badly for a frame or two - the palm swings 150 degrees and returns.
    At 60 fps that is thousands of degrees per second, which no wrist does.
    Filtering cannot remove such a frame; it only smears it over its neighbours.
    Dropping it lets the gap filling draw a clean path across instead.

    The "and comes straight back" part is what protects real motion: a rotation
    that never returns to where it started is accepted however large it is. The
    sample clip contains a genuine 180-degree wrist turn, and it survives.

    Returns the number of frames blanked.
    """
    bases = {}
    for index in range(data.shape[0]):
        hand = data[index, start:start + HAND_COUNT, :]
        if np.isfinite(hand).all():
            basis = palm_basis(hand)
            if basis is not None:
                bases[index] = basis

    measured = sorted(bases)
    rejected: list[int] = []
    position = 0
    reference = None
    while position < len(measured):
        index = measured[position]
        if reference is None or rotation_between(reference, bases[index]) <= threshold_deg:
            reference = bases[index]
            position += 1
            continue
        returned_at = None
        for look in range(1, max_run + 1):
            if position + look >= len(measured):
                break
            if rotation_between(reference, bases[measured[position + look]]) <= threshold_deg:
                returned_at = position + look
                break
        if returned_at is None:  # the hand really did move there and stay
            reference = bases[index]
            position += 1
            continue
        rejected.extend(measured[position:returned_at])
        position = returned_at

    for index in rejected:
        data[index, start:start + HAND_COUNT, :] = np.nan
    return len(rejected)


# ---------------------------------------------------------------------------
# Orientation / shape split
# ---------------------------------------------------------------------------


def axis_angle(rotation: np.ndarray) -> tuple:
    """Decompose a rotation matrix into (unit axis, angle in radians)."""
    cosine = (np.trace(rotation) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # 180 degrees: the antisymmetric part vanishes, take it from the symmetric one
        values, vectors = np.linalg.eigh(rotation)
        axis = vectors[:, int(np.argmax(values))]
        return axis / np.linalg.norm(axis), angle
    return axis / norm, angle


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for `angle` radians about a unit `axis` (Rodrigues)."""
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def interior_gaps(solve: np.ndarray, rows: slice) -> list:
    """Pairs of measured frames with untracked frames between them."""
    measured = np.flatnonzero(np.isfinite(solve[:, rows, :]).all(axis=(1, 2)))
    return [
        (int(measured[k]), int(measured[k + 1]))
        for k in range(len(measured) - 1)
        if measured[k + 1] - measured[k] > 1
    ]


def hold_handshape_in_gaps(solve: np.ndarray, blend_frames: int) -> dict:
    """Keep the last measured handshape while a hand is not tracked.

    Interpolating finger positions across a gap looks wrong, because a straight
    line between two handshapes is not a handshape. On the sample clip the right
    hand disappears for 14 frames between two quite different poses, and halfway
    through the gap the interpolated middle finger is 54% of its real length -
    the fingers appear to curl into the palm and twist back out again. None of
    that was measured; it is an artefact of averaging two different hands.

    Holding the last shape instead says only what was actually seen: the hand
    keeps the pose it had when it went out of view. The last `blend_frames` of
    the gap ease into the pose it is about to be seen in, so that tracking
    resumes without a snap - set that to 0 for a hard hold.

    The palm's *orientation* is not held; see ``slerp_orientation_gaps``. The
    hand still travels and turns to where it reappears, which is motion the arm
    is doing anyway - only the fingers stay put.

    Returns per-hand counts of the frames held.
    """
    held = {}
    for label, source in SOLVE_HAND_OFFSETS.items():
        rows = slice(source, source + HAND_COUNT)
        count = 0
        for start, end in interior_gaps(solve, rows):
            last_seen = solve[start, rows, :].copy()
            next_seen = solve[end, rows, :].copy()
            gap_length = end - start - 1
            # Never spend more than half a gap easing, so that even a short gap is
            # mostly the handshape that was actually measured.
            blend = min(blend_frames, gap_length // 2)
            for index in range(start + 1, end):
                remaining = end - index  # frames until the hand is measured again
                if blend > 0 and remaining <= blend:
                    weight = (blend - remaining + 1) / (blend + 1)
                    solve[index, rows, :] = (1.0 - weight) * last_seen + weight * next_seen
                else:
                    solve[index, rows, :] = last_seen
                count += 1
        held[label] = count
    return held


# Each pose landmark that hangs off another one. A limb that drops out is held as an
# offset from its parent, so it keeps its bone length and still travels with the joint
# above it, instead of being frozen in space while the rest of the body moves.
BODY_PARENTS = {
    13: 11, 15: 13, 17: 15, 19: 15, 21: 15,  # left arm and hand points
    14: 12, 16: 14, 18: 16, 20: 16, 22: 16,  # right arm and hand points
    25: 23, 27: 25, 29: 27, 31: 27,  # left leg and foot
    26: 24, 28: 26, 30: 28, 32: 28,  # right leg and foot
}


def _fill_order() -> list:
    """Pose landmarks, parents before children."""
    order = []

    def visit(marker: int) -> None:
        if marker in order:
            return
        if marker in BODY_PARENTS:
            visit(BODY_PARENTS[marker])
        order.append(marker)

    for marker in range(POSE_COUNT):
        visit(marker)
    return order


def body_reliability(
    frames: list[dict], threshold: float, frames_to_hold: int, frames_to_recover: int
) -> np.ndarray:
    """(n_frames, 33) flags: True where MediaPipe actually saw that body landmark.

    Pose world landmarks come with a visibility score, and it drops when a limb is
    hidden behind the body or outside the picture - exactly when MediaPipe is
    guessing. A landmark counts as lost only after `frames_to_hold` low frames in a
    row (so one noisy frame does not trip it) and as back only after
    `frames_to_recover` good ones (so it does not flicker in and out).
    """
    count = len(frames)
    low = np.ones((count, POSE_COUNT), dtype=bool)
    for index, frame in enumerate(frames):
        pose = frame.get("pose_world")
        if pose is None:
            continue
        for marker, point in enumerate(pose):
            visibility = point[3] if len(point) > 3 else 1.0
            low[index, marker] = visibility < threshold

    def runs(flags: np.ndarray) -> list:
        found, start = [], None
        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                found.append((start, index))
                start = None
        if start is not None:
            found.append((start, len(flags)))
        return found

    reliable = np.ones((count, POSE_COUNT), dtype=bool)
    for marker in range(POSE_COUNT):
        lost = np.zeros(count, dtype=bool)
        for start, end in runs(low[:, marker]):
            if end - start >= frames_to_hold:
                lost[start:end] = True
        # A short good stretch between two lost ones is not a real recovery.
        for start, end in runs(~lost):
            if start > 0 and end < count and end - start < frames_to_recover:
                lost[start:end] = True
        reliable[:, marker] = ~lost
    return reliable


def hold_body_in_gaps(data: np.ndarray, reliable: np.ndarray, blend_frames: int) -> dict:
    """Blank unreliable body landmarks and bridge each gap by holding, then easing back.

    The body counterpart of ``hold_handshape_in_gaps``. Straight-line interpolation
    across an arm or leg that left the picture sweeps it through poses nobody
    measured; holding the last seen position, then easing into the next measured
    one over the last `blend_frames` of the gap, says only what was observed.

    Limbs are held relative to their parent joint (``BODY_PARENTS``). A landmark
    that was never reliable is left as measured, since blanking it would leave
    nothing to hold. Gaps touching the clip's start or end are left to
    ``hold_edges``. Edits `data` in place and returns the frames held per landmark.
    """
    pose = data[:, :POSE_COUNT, :]
    held = {}
    for marker in range(POSE_COUNT):
        if not reliable[:, marker].any():
            continue
        pose[~reliable[:, marker], marker, :] = np.nan
    for marker in _fill_order():
        track = pose[:, marker, :]
        measured = np.flatnonzero(np.isfinite(track).all(axis=1))
        parent = BODY_PARENTS.get(marker)
        count = 0
        for k in range(len(measured) - 1):
            start, end = int(measured[k]), int(measured[k + 1])
            if end - start <= 1:
                continue
            span = slice(start, end + 1)
            relative = parent is not None and np.isfinite(pose[span, parent, :]).all()
            if relative:
                last_seen = track[start] - pose[start, parent]
                next_seen = track[end] - pose[end, parent]
            else:
                last_seen, next_seen = track[start].copy(), track[end].copy()
            blend = min(blend_frames, (end - start - 1) // 2)
            for index in range(start + 1, end):
                remaining = end - index
                if blend > 0 and remaining <= blend:
                    weight = (blend - remaining + 1) / (blend + 1)
                    value = (1.0 - weight) * last_seen + weight * next_seen
                else:
                    value = last_seen
                track[index] = value + pose[index, parent] if relative else value
                count += 1
        if count:
            held[marker] = count
    return held


def slerp_orientation_gaps(solve: np.ndarray) -> None:
    """Turn the palm along the shortest arc across a gap, not component by component.

    skellyforge interpolates each number on its own, which for the three axes of
    an orientation means the frames in between are not rotations at all - they
    shrink towards the midpoint and have to be squared up again afterwards. Over
    the sample clip's 130-degree gap that produces a hand that shrivels and
    twists. Rotating about the single axis that carries one orientation to the
    other keeps every frame in between a real orientation.
    """
    for source in SOLVE_HAND_OFFSETS.values():
        rows = slice(source + HAND_COUNT, source + HAND_COUNT + AXIS_COUNT)
        for start, end in interior_gaps(solve, rows):
            first = orthonormalise(across=solve[start, rows, :][0], along=solve[start, rows, :][1])
            last = orthonormalise(across=solve[end, rows, :][0], along=solve[end, rows, :][1])
            if first is None or last is None:
                continue
            axis, angle = axis_angle(last @ first.T)
            span = end - start
            for index in range(start + 1, end):
                fraction = (index - start) / span
                solve[index, rows, :] = (rotation_about(axis, angle * fraction) @ first).T


def limit_rotation_rate(solve: np.ndarray, fps: float, max_degrees_per_second: float) -> dict:
    """Stop a hand's orientation from turning faster than a wrist physically can.

    Where the two hands overlap, MediaPipe's reconstruction of one of them can
    jump - on the sample clip the right palm moves 129 degrees between two
    frames, which at 60 fps is nearly 8000 degrees per second. The spike check
    above deliberately leaves it alone, because unlike a spike it does not come
    back: the hand really is turned over afterwards, and rejecting every frame
    from there on would throw away the rest of the clip.

    So the jump is followed rather than deleted - just no faster than the limit,
    which spreads it over the handful of frames a real wrist would have taken.
    Genuine signing motion stays untouched as long as the limit is set above what
    a wrist can do; it exists to catch the physically impossible, not to slow
    things down.

    Returns per-hand counts of how many frames were clamped.
    """
    limit_per_frame = math.radians(max_degrees_per_second) / fps
    clamped = {}
    for label, source in SOLVE_HAND_OFFSETS.items():
        axes_start = source + HAND_COUNT
        current = None
        count = 0
        for index in range(solve.shape[0]):
            axes = solve[index, axes_start:axes_start + AXIS_COUNT, :]
            if not np.isfinite(axes).all():
                continue
            measured = orthonormalise(across=axes[0], along=axes[1])
            if measured is None:
                continue
            if current is None:
                current = measured
            else:
                axis, angle = axis_angle(measured @ current.T)
                if angle > limit_per_frame:
                    current = rotation_about(axis, limit_per_frame) @ current
                    count += 1
                else:
                    current = measured
            solve[index, axes_start:axes_start + AXIS_COUNT, :] = current.T
        clamped[label] = count
    return clamped


def to_solve_space(data: np.ndarray) -> np.ndarray:
    """Rewrite each hand as (palm-local finger shape, palm basis axes).

    Both parts vary smoothly even when the wrist turns right over, which is what
    makes them safe to interpolate and low-pass filter. Filtering the raw
    wrist-relative coordinates instead would shrink the hand towards its wrist
    every time it rotates quickly.
    """
    frame_count = data.shape[0]
    solve = np.full((frame_count, SOLVE_MARKER_COUNT, 3), np.nan, dtype=float)
    solve[:, :POSE_COUNT, :] = data[:, :POSE_COUNT, :]
    for label, source in HAND_OFFSETS.items():
        target = SOLVE_HAND_OFFSETS[label]
        for index in range(frame_count):
            hand = data[index, source:source + HAND_COUNT, :]
            if not np.isfinite(hand).all():
                continue
            basis = palm_basis(hand)
            if basis is None:
                continue
            solve[index, target:target + HAND_COUNT, :] = (hand - hand[HAND_WRIST]) @ basis
            solve[index, target + HAND_COUNT:target + HAND_COUNT + AXIS_COUNT, :] = basis.T
    return solve


def from_solve_space(solve: np.ndarray) -> np.ndarray:
    """Rebuild wrist-relative hand landmarks from the filtered shape and axes.

    The filtered axes are no longer exactly orthonormal - each was smoothed on
    its own - so the basis is squared up again before the hand is rebuilt.
    """
    frame_count = solve.shape[0]
    data = np.full((frame_count, MARKER_COUNT, 3), np.nan, dtype=float)
    data[:, :POSE_COUNT, :] = solve[:, :POSE_COUNT, :]
    for label, target in HAND_OFFSETS.items():
        source = SOLVE_HAND_OFFSETS[label]
        for index in range(frame_count):
            shape = solve[index, source:source + HAND_COUNT, :]
            axes = solve[index, source + HAND_COUNT:source + HAND_COUNT + AXIS_COUNT, :]
            if not np.isfinite(shape).all() or not np.isfinite(axes).all():
                continue
            basis = orthonormalise(across=axes[0], along=axes[1])
            if basis is None:
                continue
            data[index, target:target + HAND_COUNT, :] = shape @ basis.T
    return data


# ---------------------------------------------------------------------------
# The skellyforge chain
# ---------------------------------------------------------------------------


def hold_edges(data: np.ndarray) -> None:
    """Extend the first and last measured frame of each marker outwards, in place.

    Interpolation cannot fill a gap with measurements on only one side, and
    skellyforge's fallback for what is left replaces it with the mean of that
    marker's x, y and z together - one number for all three axes, which
    collapses the pose. Holding the nearest measured frame instead means a clip
    that opens before the hands are visible starts on the first handshape that
    was actually measured.
    """
    for marker in range(data.shape[1]):
        track = data[:, marker, :]
        measured = np.flatnonzero(np.isfinite(track).all(axis=1))
        if measured.size == 0:
            continue
        track[: measured[0]] = track[measured[0]]
        track[measured[-1] + 1:] = track[measured[-1]]


def gap_report(data: np.ndarray, start: int, count: int) -> dict:
    """How many frames of a marker group were measured, and the longest gap in it."""
    present = np.isfinite(data[:, start:start + count, :]).all(axis=(1, 2))
    longest = current = 0
    for measured in present:
        current = 0 if measured else current + 1
        longest = max(longest, current)
    return {"measured_frames": int(present.sum()), "longest_gap_frames": int(longest)}


def shape_rows() -> np.ndarray:
    """Mask of the palm-local finger-shape markers, one flag per solve-space row.

    These are the rows that describe how the fingers are folded. Everything else
    - the body, and each hand's three palm axes - is orientation and position.
    """
    mask = np.zeros(SOLVE_MARKER_COUNT, dtype=bool)
    for source in SOLVE_HAND_OFFSETS.values():
        mask[source:source + HAND_COUNT] = True
    return mask


def post_process(
    data: np.ndarray, fps: float, args: argparse.Namespace, skip_filter: np.ndarray | None = None
) -> np.ndarray:
    """Run freemocap's interpolate-then-filter chain over a marker array.

    `skip_filter` marks rows that are interpolated but not filtered. Stage 2.5
    uses it to keep the Butterworth off the finger shape, because a One Euro
    filter runs over the joint angles later and two low-passes fighting over the
    same signal cannot be tuned independently - lower the cutoff to kill jitter
    and the other one's lag compounds. The palm axes stay filtered here: they
    are not joint angles and stage 2.5 leaves them alone.
    """
    skellyforge = import_skellyforge()
    frame_count = data.shape[0]

    # Markers never seen at all stay NaN through interpolation; keep them out of
    # what skellyforge sees so its nanmean fallback is not asked for the mean of
    # nothing, then put the empty column back afterwards.
    never_measured = ~np.isfinite(data).all(axis=2).any(axis=0)
    working = data[:, ~never_measured, :]
    if never_measured.any():
        print(f"[postprocess] {int(never_measured.sum())} markers were never detected and are left empty")
    if working.shape[1] == 0:
        return data

    trajectory = skellyforge["Trajectory3d"](
        start_frame=0,
        end_frame=frame_count,
        triangulated_data=working,
        reprojection_error=np.zeros(working.shape[:2]),
        reprojection_error_by_camera=np.zeros((1, *working.shape[:2])),
    )

    if not args.no_interpolate:
        trajectory = skellyforge["interpolate_trajectory"](
            trajectory=trajectory, config=skellyforge["InterpolationConfig"]()
        )
        print("[postprocess] interpolated gaps (skellyforge, linear)")

    interpolated = trajectory.triangulated_data

    if not args.no_filter:
        # filtfilt pads the signal by three times the filter length at each end.
        minimum_frames = 3 * (2 * args.order + 1)
        if frame_count <= minimum_frames:
            print(
                f"[postprocess] only {frame_count} frames - too short for an order-{args.order} "
                f"filter (needs more than {minimum_frames}); skipping the filter"
            )
        else:
            # Which of the surviving columns to filter. `skip_filter` indexes the
            # full solve space, so it has to be narrowed the same way the data was.
            filtered_columns = (
                np.ones(interpolated.shape[1], dtype=bool)
                if skip_filter is None
                else ~skip_filter[~never_measured]
            )
            if filtered_columns.any():
                subset = skellyforge["Trajectory3d"](
                    start_frame=0,
                    end_frame=frame_count,
                    triangulated_data=interpolated[:, filtered_columns, :],
                    reprojection_error=np.zeros(interpolated[:, filtered_columns, :].shape[:2]),
                    reprojection_error_by_camera=np.zeros(
                        (1, *interpolated[:, filtered_columns, :].shape[:2])
                    ),
                )
                subset = skellyforge["filter_trajectory"](
                    trajectory=subset,
                    config=skellyforge["FilterConfig"](
                        cutoff=args.cutoff, sampling_rate=fps, order=args.order
                    ),
                )
                interpolated = interpolated.copy()
                interpolated[:, filtered_columns, :] = subset.triangulated_data
            held_back = int((~filtered_columns).sum())
            print(
                f"[postprocess] filtered (skellyforge butter_low_pass, "
                f"{args.cutoff} Hz cutoff, order {args.order}, {fps:.2f} Hz sampling)"
                + (
                    f" - {held_back} finger-shape markers left for stage 2.5's One Euro filter"
                    if held_back
                    else ""
                )
            )

    result = np.full_like(data, np.nan)
    result[:, ~never_measured, :] = interpolated
    return result


def skellyforge_version() -> str:
    try:
        from importlib.metadata import version

        return version("skellyforge")
    except Exception:  # noqa: BLE001 - version reporting must never fail the run
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    landmarks_path = args.landmarks.expanduser().resolve()
    if not landmarks_path.is_file():
        raise SystemExit(f"Landmark file not found: {landmarks_path}")
    data_in = json.loads(landmarks_path.read_text(encoding="utf-8"))
    frames = data_in.get("frames")
    if not frames:
        raise SystemExit(f"{landmarks_path} contains no frames.")

    fps = args.fps or data_in.get("fps") or 30.0
    if args.fps and data_in.get("fps") and abs(args.fps - float(data_in["fps"])) > 1e-6:
        print(
            f"[postprocess] warning: --fps {args.fps:g} overrides the {float(data_in['fps']):g} fps "
            f"recorded in {landmarks_path.name}; the output header will carry {args.fps:g}"
        )

    # No trimming: every frame of the input video is processed.
    trimmed = {"dropped_before": 0, "dropped_after": 0}

    # The confidence gate, run before anything touches the data. Hands it
    # rejects are blanked so that the filter never sees them; the gap they leave
    # is crossed by the hold-and-slerp machinery further down, and stage 2.5
    # replaces that hold with a proper extrapolation.
    reliable = None
    gated = {}
    if args.confidence_gate:
        config = hand_stabilizer.load_config(args.stabilize_config)
        scores = hand_stabilizer.confidence_track(frames, config)
        reliable = {
            label: hand_stabilizer.gate_track(scores[label]["hand"], config)
            for label in HAND_OFFSETS
        }
        for label in HAND_OFFSETS:
            detected = sum(1 for frame in frames if label in (frame.get("hands") or {}))
            kept = sum(
                1
                for index, frame in enumerate(frames)
                if label in (frame.get("hands") or {}) and reliable[label][index]
            )
            gated[label] = detected - kept
            if gated[label]:
                print(
                    f"[postprocess] {label}: confidence gate rejected {gated[label]} of "
                    f"{detected} detected frames before filtering"
                )

    raw = build_marker_array(frames, reliable)

    before = {
        "pose": gap_report(raw, 0, POSE_COUNT),
        "Left": gap_report(raw, HAND_OFFSETS["Left"], HAND_COUNT),
        "Right": gap_report(raw, HAND_OFFSETS["Right"], HAND_COUNT),
    }
    for group, report in before.items():
        gap_seconds = report["longest_gap_frames"] / fps if fps else 0.0
        print(
            f"[postprocess] {group}: measured on {report['measured_frames']}/{len(frames)} frames, "
            f"longest gap {report['longest_gap_frames']} frames ({gap_seconds:.2f} s)"
        )

    spikes = {}
    if args.spike_threshold > 0:
        for label, offset in HAND_OFFSETS.items():
            spikes[label] = reject_orientation_spikes(raw, offset, args.spike_threshold, args.max_spike_run)
            if spikes[label]:
                print(
                    f"[postprocess] {label}: dropped {spikes[label]} frames whose palm orientation "
                    f"jumped more than {args.spike_threshold:.0f} degrees and came straight back"
                )

    body_held = {}
    if args.body_visibility_threshold > 0 and not args.no_interpolate:
        body_ok = body_reliability(
            frames, args.body_visibility_threshold, args.body_frames_to_hold, args.body_frames_to_recover
        )
        body_frames = int(round(max(0.0, args.body_return_blend) * fps))
        body_held = hold_body_in_gaps(raw, body_ok, body_frames)
        if body_held:
            names = data_in.get("pose_landmark_names") or []
            summary = ", ".join(
                f"{names[m] if m < len(names) else m}={n}" for m, n in sorted(body_held.items())
            )
            print(
                f"[postprocess] body: held {sum(body_held.values())} low-visibility landmark frames "
                f"(easing back over up to {body_frames} frames per gap): {summary}"
            )

    solve = to_solve_space(raw)
    held = {}
    if not args.no_interpolate:
        # Fill the hand gaps here, in a space where a hand stays a hand, before
        # skellyforge's straight-line interpolation gets to them.
        slerp_orientation_gaps(solve)
        if args.handshape_in_gaps == "hold":
            blend_frames = int(round(max(0.0, args.gap_return_blend) * fps))
            held = hold_handshape_in_gaps(solve, blend_frames)
            for label, count in held.items():
                if count:
                    print(
                        f"[postprocess] {label}: held the last measured handshape for {count} "
                        f"untracked frames, easing back over up to {blend_frames} frames per gap"
                    )
    hold_edges(solve)
    solve = post_process(
        solve, fps, args, shape_rows() if args.hand_shape_filter == "none" else None
    )

    clamped = {}
    if args.max_wrist_speed > 0:
        clamped = limit_rotation_rate(solve, fps, args.max_wrist_speed)
        for label, count in clamped.items():
            if count:
                print(
                    f"[postprocess] {label}: held {count} frames to the "
                    f"{args.max_wrist_speed:.0f} deg/s wrist limit"
                )
    processed = from_solve_space(solve)

    data_out = dict(data_in)
    # Later stages (stabilize_hands.py, the retarget) read the top-level fps, so
    # an explicit override has to land there too, not only in the filter record -
    # otherwise the file is filtered at one rate and replayed at another.
    data_out["fps"] = fps
    data_out["frames"] = frames_from_array(processed, frames, reliable)
    data_out["frame_count"] = len(frames)
    data_out["post_processing"] = {
        "library": "skellyforge (freemocap)",
        "version": skellyforge_version(),
        "interpolation": None if args.no_interpolate else "linear",
        "filter": None
        if args.no_filter
        else {"method": "butter_low_pass", "cutoff_hz": args.cutoff, "order": args.order, "sampling_rate_hz": fps},
        "hands_filtered_as": "palm orientation plus palm-local finger shape",
        "source": landmarks_path.name,
        "measured_before": before,
        "spike_frames_dropped": spikes,
        "trim": trimmed,
        "handshape_in_gaps": args.handshape_in_gaps,
        "handshape_held_frames": held,
        "body_hold": {
            "visibility_threshold": args.body_visibility_threshold,
            "frames_to_hold": args.body_frames_to_hold,
            "frames_to_recover": args.body_frames_to_recover,
            "return_blend_seconds": args.body_return_blend,
            "held_frames_by_landmark_index": body_held,
        },
        "confidence_gate": {
            "applied": bool(args.confidence_gate),
            "rejected_frames": gated,
            "note": (
                "Hands rejected here were detected by MediaPipe but scored as unreliable, "
                "and were blanked before the Butterworth filter so that filtfilt could not "
                "spread them into neighbouring frames."
            ),
        },
        "hand_shape_filter": args.hand_shape_filter,
        "rate_limited_frames": clamped,
        "max_wrist_speed_deg_per_second": args.max_wrist_speed,
        "note": (
            "Frames outside 'hands_measured' were filled by interpolation, not observed. "
            "MediaPipe visibility scores are dropped here."
        ),
    }

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data_out), encoding="utf-8")
    print(f"[postprocess] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
