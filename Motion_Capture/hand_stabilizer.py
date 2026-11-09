"""Occlusion handling and stabilisation for MediaPipe hand tracking.

This is the library behind stage 2.5. It is imported by three callers, which is
why it is a module rather than a script:

  * ``extract_landmarks.py``  (stage 1)   - nothing, currently: stage 1 only
    writes the raw signals this module later turns into a confidence score.
  * ``postprocess_landmarks.py`` (stage 2) - :func:`confidence_track` and
    :func:`gate_track`, to blank unreliable hands *before* the Butterworth
    filter sees them. That ordering matters: skellyforge's filter is
    ``filtfilt``, which smears a bad frame both forwards and backwards in time,
    so a garbage frame that survives into stage 2 contaminates its neighbours.
  * ``stabilize_hands.py`` (stage 2.5)   - everything else.

Pure numpy plus PyYAML, so it imports in both interpreters this pipeline uses
(``.venv311`` for mediapipe, ``.venv`` for skellyforge).

Why any of this is needed
-------------------------
MediaPipe's hand solution reports ``visibility`` and ``presence`` as exactly
0.0 on every landmark, image-space and world - measured, not assumed. There is
no confidence score to threshold, so :func:`confidence_components` derives one
from four things that *are* measurable, the strongest being that bones do not
change length.

The pipeline, per finger, per frame::

    confidence -> hysteresis -> hold/extrapolate/decay -> blend back
               -> anatomical clamp -> one euro -> forward kinematics -> landmarks

Everything tunable lives in ``stabilize.yaml``; this file reads it and contains
no thresholds of its own.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent / "stabilize.yaml"

HAND_COUNT = 21
HAND_WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
HAND_LABELS = ("Left", "Right")

# A hand's chirality decides which way its palm normal points, and therefore the
# sign of "curls towards the palm".
#
# Derive it from something checkable rather than from a mental rotation, which is
# easy to get backwards: lay your right hand palm-down with the fingers pointing
# away from you, and the thumb is on your LEFT. In a right-handed frame with
# +x east, +y north, +z up, that is fingers along +y, thumb and index to the
# west, palm facing down. So
#
#     (pinky->index) x (wrist->middle) = (-x) x (+y) = -z = DOWN = the palm side.
#
# For the right hand that cross product points out through the *palm*, not the
# back of the hand, so the dorsal normal is its negative. The left hand mirrors.
#
# Getting this backwards is not a cosmetic error: every curled finger reads as
# hyperextension, and the joint limits then clamp real signing away as impossible.
# ``test_a_curled_finger_on_a_real_right_hand_reads_as_flexion`` pins it against
# the anatomy above rather than against this module's own conventions.
CHIRALITY = {"Right": -1.0, "Left": 1.0}

# Each finger as (root, proximal, middle, distal) landmark indices. The joints
# are at the first three: a 2-DOF one at the root and two hinges after it.
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
ANGLE_NAMES = ("mcp_flexion", "mcp_abduction", "pip_flexion", "dip_flexion")
ANGLE_COUNT = len(ANGLE_NAMES)

# The landmarks that make up the rigid palm. These are not driven by joint
# angles - they are carried across as one solid piece.
PALM_LANDMARKS = (0, 1, 5, 9, 13, 17)

# The 20 bone segments: for each finger, the wrist to its root landmark and then
# the three phalanges. Grouped four per finger in FINGER_ORDER, so that one
# finger's rigidity can be scored from a contiguous slice.
SEGMENTS = tuple(
    pair
    for finger in FINGER_ORDER
    for chain in (FINGER_CHAINS[finger],)
    for pair in (
        (HAND_WRIST, chain[0]),
        (chain[0], chain[1]),
        (chain[1], chain[2]),
        (chain[2], chain[3]),
    )
)
SEGMENT_COUNT = len(SEGMENTS)
FINGER_SEGMENTS = {finger: slice(4 * i, 4 * i + 4) for i, finger in enumerate(FINGER_ORDER)}

LIVE, HOLD, BLEND = "live", "hold", "blend"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path | str | None = None) -> dict:
    """Read stabilize.yaml. Every tunable in this module comes from here."""
    import yaml

    config_path = Path(path) if path is not None else CONFIG_PATH
    if not config_path.is_file():
        raise SystemExit(f"Stabiliser config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} did not parse as a mapping.")
    return config


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-9:
        return None
    return vector / length


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for `angle` radians about a unit `axis`."""
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def signed_angle(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float:
    """Angle from `first` to `second`, signed about `axis`, in radians."""
    return math.atan2(float(np.dot(np.cross(first, second), axis)), float(np.dot(first, second)))


def palm_frame(hand: np.ndarray) -> np.ndarray | None:
    """Orthonormal palm basis, columns (X across knuckles, Y along fingers, Z normal).

    Deliberately the same construction as ``postprocess_landmarks.palm_basis``,
    so the two stages agree about what "the palm's orientation" means.
    """
    along = _unit(hand[MIDDLE_MCP] - hand[HAND_WRIST])
    if along is None:
        return None
    across = hand[INDEX_MCP] - hand[PINKY_MCP]
    across = _unit(across - along * float(np.dot(across, along)))
    if across is None:
        return None
    return np.stack([across, along, np.cross(across, along)], axis=1)


def anatomical_axes(chirality: float) -> tuple:
    """The frame joint angles are measured in, expressed in palm-local coordinates.

    In palm-local coordinates the palm basis is the identity by construction, so
    these are just the axes, mirrored for the left hand:

      Ya  along the fingers, Za  out through the back of the hand,
      Xa  Ya x Za, which completes a right-handed frame.

    Because Za flips between hands, so does Xa - which means raw abduction is
    mirrored too. :func:`hand_to_angles` multiplies it back by the chirality so
    that the limits in stabilize.yaml read anatomically for both hands.
    """
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, chirality])
    return np.cross(y_axis, z_axis), y_axis, z_axis


# ---------------------------------------------------------------------------
# 1. Confidence
# ---------------------------------------------------------------------------


def segment_lengths(hand: np.ndarray) -> np.ndarray:
    """The 20 bone lengths of one hand, in metres, grouped four per finger."""
    return np.array([float(np.linalg.norm(hand[b] - hand[a])) for a, b in SEGMENTS])


def palm_facing(hand: np.ndarray) -> float:
    """|palm normal . camera axis|: 1.0 square to the camera, 0.0 edge-on.

    A palm seen edge-on gives monocular depth almost nothing to work with, which
    is the case that makes fingers snap into impossible bends. MediaPipe's world
    landmarks keep the camera's axes, so +Z is simply the view direction.
    """
    first = hand[INDEX_MCP] - hand[HAND_WRIST]
    second = hand[PINKY_MCP] - hand[HAND_WRIST]
    normal = _unit(np.cross(first, second))
    if normal is None:
        return 0.0
    return abs(float(normal[2]))


def _score(value: float, good: float, bad: float) -> float:
    """Map a raw signal onto 0..1, linearly between its `good` and `bad` points."""
    if not math.isfinite(value):
        return 0.0
    if good == bad:
        return 1.0 if value == good else 0.0
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def rigidity_residual(lengths: np.ndarray, reference: np.ndarray, rows: slice | None = None) -> float:
    """How far this frame's bone lengths sit from that hand's own median, 0 = perfect.

    Length-weighted, and deliberately so. The three shortest segments (the
    proximal thumb and the pinky and middle knuckles, 7-9 mm on the sample clip)
    show 17-44% interquartile spread purely because they are short, while the
    long well-defined segments sit at 4-15%. An unweighted mean would be
    dominated by the segments that carry the least information.
    """
    if rows is not None:
        lengths, reference = lengths[rows], reference[rows]
    usable = np.isfinite(lengths) & np.isfinite(reference) & (reference > 1e-6)
    if not usable.any():
        return float("inf")
    deviation = np.abs(lengths[usable] - reference[usable]) / reference[usable]
    weights = reference[usable]
    order = np.argsort(deviation)
    deviation, weights = deviation[order], weights[order]
    # Weighted median: the deviation at which half the total bone length is below.
    cumulative = np.cumsum(weights)
    return float(deviation[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def confidence_components(
    hand: np.ndarray | None,
    raw: dict | None,
) -> dict:
    """Gather the four raw confidence signals for one hand on one frame.

    `raw` is the block stage 1 wrote (out-of-frame fraction, handedness score,
    and the segment lengths and palm facing measured *before* any smoothing).
    Preferring stage 1's numbers matters: by the time stage 2.5 runs, the
    Butterworth filter has already made the geometry look more plausible than
    the tracker actually reported it, so measuring rigidity here would
    under-report exactly the frames worth catching.
    """
    components = {
        "out_of_frame": 1.0,
        "handedness_score": 0.5,
        "palm_facing": 0.0,
        "lengths": np.full(SEGMENT_COUNT, np.nan),
        "measured": False,
    }
    if raw:
        components["out_of_frame"] = float(raw.get("out_of_frame", 1.0))
        components["handedness_score"] = float(raw.get("handedness_score", 0.5))
        components["palm_facing"] = float(raw.get("palm_facing", 0.0))
        lengths = raw.get("segment_lengths")
        if lengths is not None and len(lengths) == SEGMENT_COUNT:
            components["lengths"] = np.asarray(lengths, dtype=float)
        components["measured"] = True
    if hand is not None and np.isfinite(hand).all() and not np.isfinite(components["lengths"]).any():
        # Fall back to the geometry in front of us when stage 1 wrote no block,
        # which means an older landmarks.json. Callers must NOT route a hand
        # that stage 2 synthesised through here: a held handshape has perfectly
        # plausible bone lengths and would score as well-tracked, which is the
        # opposite of the truth. :func:`confidence_track` gates on
        # ``hands_reliable`` before it gets this far.
        components["palm_facing"] = palm_facing(hand)
        components["lengths"] = segment_lengths(hand)
    return components


def combine_confidence(components: dict, reference: np.ndarray, config: dict, rows: slice | None = None) -> tuple:
    """Blend the raw signals into one 0..1 score. Returns (score, residual)."""
    settings = config["confidence"]
    weights, scales = settings["weights"], settings["scales"]
    residual = rigidity_residual(components["lengths"], reference, rows)
    parts = {
        "rigidity": _score(residual, scales["rigidity_good"], scales["rigidity_bad"]),
        "out_of_frame": _score(
            components["out_of_frame"], scales["out_of_frame_good"], scales["out_of_frame_bad"]
        ),
        "palm_facing": _score(
            components["palm_facing"], scales["palm_facing_good"], scales["palm_facing_bad"]
        ),
        "handedness": _score(
            components["handedness_score"], scales["handedness_good"], scales["handedness_bad"]
        ),
    }
    total = sum(weights[name] for name in parts)
    score = sum(weights[name] * value for name, value in parts.items()) / (total or 1.0)
    return float(score), residual


def reference_lengths(component_track: list, config: dict) -> np.ndarray:
    """Each hand's own median bone lengths, from its better-tracked frames.

    Per hand rather than per person: MediaPipe's two hands are separate
    detections and their scale estimates do not have to agree.
    """
    stacked = np.array(
        [c["lengths"] for c in component_track if c is not None and np.isfinite(c["lengths"]).all()]
    )
    if stacked.size == 0:
        return np.full(SEGMENT_COUNT, np.nan)
    # A first pass with the raw median, then a second using only the frames that
    # agreed with it, so a run of badly-solved frames cannot drag the reference.
    coarse = np.median(stacked, axis=0)
    residuals = np.array([rigidity_residual(row, coarse) for row in stacked])
    keep = residuals <= config["confidence"]["scales"]["rigidity_bad"]
    if keep.sum() >= config["reconstruct"]["min_geometry_frames"]:
        return np.median(stacked[keep], axis=0)
    return coarse


def confidence_track(frames: list, config: dict) -> dict:
    """Per-hand, per-frame confidence for a whole clip.

    Returns ``{label: {"hand": [...], "fingers": [[...]], "residual": [...],
    "components": [...], "reference": array}}``.
    """
    result = {}
    for label in HAND_LABELS:
        components = []
        for frame in frames:
            # A hand that a previous stage filled in rather than measured gets no
            # confidence at all, however plausible its geometry looks - holding a
            # handshape produces perfect bone lengths, and scoring those would
            # hand a synthesised frame the score of a well-tracked one.
            observed = frame.get("hands_reliable")
            if observed is None:
                observed = frame.get("hands_measured")
            if observed is not None and label not in observed:
                components.append(confidence_components(None, None))
                continue
            points = (frame.get("hands") or {}).get(label)
            hand = np.asarray(points, dtype=float) if points is not None else None
            raw = (frame.get("hand_confidence") or {}).get(label)
            components.append(confidence_components(hand, raw))
        reference = reference_lengths(components, config)
        hand_scores, finger_scores, residuals = [], [], []
        for entry in components:
            score, residual = combine_confidence(entry, reference, config)
            hand_scores.append(score)
            residuals.append(residual)
            finger_scores.append(
                [
                    combine_confidence(entry, reference, config, FINGER_SEGMENTS[finger])[0]
                    for finger in FINGER_ORDER
                ]
            )
        result[label] = {
            "hand": hand_scores,
            "fingers": finger_scores,
            "residual": residuals,
            "components": components,
            "reference": reference,
        }
    return result


# ---------------------------------------------------------------------------
# 2. Hysteresis
# ---------------------------------------------------------------------------


class HysteresisGate:
    """Turn a noisy per-frame confidence into a stable reliable/unreliable flag.

    A single bad frame must not trip the hold, and a single good frame must not
    end it - otherwise the output flickers between live and held, which reads
    worse than either. So a switch needs `frames_to_hold` consecutive bad frames
    or `frames_to_release` consecutive good ones.
    """

    def __init__(self, config: dict):
        settings = config["hysteresis"]
        self.threshold = float(config["confidence"]["threshold"])
        self.to_hold = int(settings["frames_to_hold"])
        self.to_release = int(settings["frames_to_release"])
        # Undecided until the first frame arrives. Starting out trusted would let
        # the opening `frames_to_hold - 1` frames through however bad they were:
        # the hysteresis exists to smooth over a change of state, and on the
        # first frame there is no earlier state to smooth over.
        self.reliable: bool | None = None
        self._low_run = 0
        self._good_run = 0

    def update(self, score: float) -> bool:
        """Feed one frame's confidence; returns whether to trust live data now."""
        if score < self.threshold:
            self._low_run += 1
            self._good_run = 0
        else:
            self._good_run += 1
            self._low_run = 0
        if self.reliable is None:
            self.reliable = score >= self.threshold
        elif self.reliable and self._low_run >= self.to_hold:
            self.reliable = False
        elif not self.reliable and self._good_run >= self.to_release:
            self.reliable = True
        return self.reliable


def gate_track(scores: list, config: dict) -> list:
    """Run :class:`HysteresisGate` over a whole clip. Used by stage 2's pre-filter gate."""
    gate = HysteresisGate(config)
    return [gate.update(score) for score in scores]


# ---------------------------------------------------------------------------
# 3. Joint angles: measurement and forward kinematics
# ---------------------------------------------------------------------------


def hand_to_angles(
    hand: np.ndarray, chirality: float, abduction_offsets: np.ndarray | None = None
) -> np.ndarray | None:
    """Measure 5x4 joint angles, in degrees, from one hand's 21 landmarks.

    The hand is taken into its own palm frame first, so the angles describe how
    the fingers are folded and nothing about where the hand is or which way it
    points. That separation is what lets a held finger ride a live wrist: the
    angles are frozen, the palm frame is not.

    Per finger: the root joint has two degrees of freedom (flexion out of the
    palm plane, abduction within it) and the two joints after it are hinges
    about a single axis carried down the chain, which is what an anatomical
    finger actually does.

    `abduction_offsets` is one angle per finger, in degrees, subtracted from the
    measured abduction. Without it, spread is measured from the middle-finger
    metacarpal axis, and the knuckle arch then reads as a permanent splay - on
    real clips the median ran to -25 degrees, far enough that a finger sat
    outside its own limit while resting. :func:`median_geometry` measures the
    offsets from the clip so that zero means "this hand's own neutral".
    """
    basis = palm_frame(hand)
    if basis is None:
        return None
    local = (hand - hand[HAND_WRIST]) @ basis
    x_axis, y_axis, z_axis = anatomical_axes(chirality)

    angles = np.full((len(FINGER_ORDER), ANGLE_COUNT), np.nan)
    for index, finger in enumerate(FINGER_ORDER):
        root, proximal, middle, distal = FINGER_CHAINS[finger]
        first = _unit(local[proximal] - local[root])
        second = _unit(local[middle] - local[proximal])
        third = _unit(local[distal] - local[middle])
        if first is None or second is None or third is None:
            continue

        # Root joint. Decomposed so that abduction is read first and flexion
        # second: the singular case is then abduction at +-90 degrees, which no
        # finger reaches, rather than flexion at 90, which they all do.
        abduction = math.asin(float(np.clip(np.dot(first, x_axis), -1.0, 1.0)))
        flexion = math.atan2(-float(np.dot(first, z_axis)), float(np.dot(first, y_axis)))

        rotation = rotation_about(x_axis, -flexion) @ rotation_about(z_axis, -abduction)
        hinge = rotation @ x_axis
        spread = math.degrees(abduction) * chirality  # report anatomically, not frame-wise
        if abduction_offsets is not None:
            spread -= float(abduction_offsets[index])
        angles[index] = (
            math.degrees(flexion),
            spread,
            math.degrees(-signed_angle(first, second, hinge)),
            math.degrees(-signed_angle(second, third, hinge)),
        )
    return angles


def angles_to_hand(
    angles: np.ndarray,
    palm_local: np.ndarray,
    bone_lengths: dict,
    basis: np.ndarray,
    chirality: float,
    abduction_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild 21 wrist-relative landmarks from joint angles by forward kinematics.

    `palm_local` is the hand's median palm shape and `bone_lengths` its median
    bone lengths, so the result cannot have a stretched finger or a palm that
    changes size - which removes a large part of what "anatomically impossible"
    looked like, quite separately from the joint limits.

    `basis` is the *live* palm orientation for this frame. Held fingers
    therefore move rigidly with the tracked hand while their internal angles
    stay put, which is the behaviour the whole hold exists to produce.
    """
    x_axis, y_axis, z_axis = anatomical_axes(chirality)
    local = np.zeros((HAND_COUNT, 3))
    for slot, landmark in enumerate(PALM_LANDMARKS):
        local[landmark] = palm_local[slot]

    for index, finger in enumerate(FINGER_ORDER):
        root, proximal, middle, distal = FINGER_CHAINS[finger]
        spread = angles[index][1]
        if abduction_offsets is not None:
            spread += float(abduction_offsets[index])  # undo the neutral calibration
        flexion, _, pip, dip = np.radians(angles[index])
        abduction = math.radians(spread) * chirality  # back into frame coordinates

        rotation = rotation_about(x_axis, -flexion) @ rotation_about(z_axis, -abduction)
        first = rotation @ y_axis
        hinge = rotation @ x_axis
        second = rotation_about(hinge, -pip) @ first
        third = rotation_about(hinge, -dip) @ second

        lengths = bone_lengths[finger]
        local[proximal] = local[root] + lengths[0] * first
        local[middle] = local[proximal] + lengths[1] * second
        local[distal] = local[middle] + lengths[2] * third

    return local @ basis.T


def median_geometry(hands: list, scores: list, chirality: float, config: dict) -> dict | None:
    """A hand's median palm shape, bone lengths and neutral finger spread.

    Everything the reconstruction needs that is *not* a joint angle. Taken from
    the frames the confidence score likes, falling back to all of them rather
    than failing, because a clip where nothing clears the bar still has to
    produce something.

    The abduction offsets are the per-finger median spread, and they are what
    makes the abduction limits mean anything. Measured from the middle-finger
    metacarpal axis, the knuckle arch gives every other finger a large constant
    splay - enough that a resting middle finger sat outside its own +-15 degree
    limit on one clip, and abduction was being clamped on a third of all
    angle-frames. Subtracting the median moves the zero to this hand's own
    neutral.

    The offsets are this *clip's* neutral, not an anatomical one: on a clip that
    is mostly one handshape they describe that handshape's splay. That is still
    a far better zero than assuming every finger points along the middle
    metacarpal, but it means the offsets are not a measurement of the signer.
    """
    settings = config["reconstruct"]
    minimum = int(settings["min_geometry_frames"])
    threshold = float(settings["geometry_from_confidence"])

    usable = [
        index
        for index, hand in enumerate(hands)
        if hand is not None and np.isfinite(hand).all() and palm_frame(hand) is not None
    ]
    if not usable:
        return None
    chosen = [index for index in usable if scores[index] >= threshold]
    confident = len(chosen) >= minimum
    if not confident:
        chosen = usable

    locals_ = []
    for index in chosen:
        hand = hands[index]
        basis = palm_frame(hand)
        locals_.append((hand - hand[HAND_WRIST]) @ basis)
    stacked = np.median(np.array(locals_), axis=0)

    # Square the median palm up again: a per-coordinate median of many frames is
    # not exactly aligned to the palm axes, and the forward kinematics assumes it is.
    settled = palm_frame(stacked)
    if settled is None:
        return None
    stacked = (stacked - stacked[HAND_WRIST]) @ settled

    # Per-finger neutral spread, measured per frame and then taken as a median so
    # that a few badly-solved frames cannot move the zero.
    spreads = []
    for index in chosen:
        measured = hand_to_angles(hands[index], chirality)
        if measured is not None and np.isfinite(measured).all():
            spreads.append(measured[:, 1])
    offsets = (
        np.median(np.array(spreads), axis=0)
        if len(spreads) >= config["reconstruct"]["min_geometry_frames"]
        else np.zeros(len(FINGER_ORDER))
    )
    # A neutral cannot sit outside the anatomical range. On a hand that is badly
    # tracked for most of a clip the median pose is itself nonsense - the worst
    # case here produced a 56 degree "resting" splay - and moving the zero there
    # would bless that nonsense rather than correct for the knuckle arch.
    limits = limit_array(config)[:, 1, :]
    offsets = np.clip(offsets, limits[:, 0], limits[:, 1])

    # Bone lengths are the median of each frame's own lengths, NOT the lengths
    # of the median pose above. The palm is rigid, so its per-coordinate median
    # is still a palm; a finger is not, and the median of a finger that opens
    # and closes across the clip has its joints pulled towards one another - a
    # middle phalanx came out at roughly half its length. Rebuilding on those
    # would ship stubby fingers.
    lengths = reference_lengths(
        [{"lengths": segment_lengths(hands[index])} for index in chosen], config
    )
    # Each finger's first segment is wrist-to-root, which the palm shape covers.
    bone_lengths = {
        finger: [float(value) for value in lengths[FINGER_SEGMENTS[finger]][1:]]
        for finger in FINGER_ORDER
    }

    return {
        "abduction_offsets": offsets,
        "palm_local": np.array([stacked[landmark] for landmark in PALM_LANDMARKS]),
        "bone_lengths": bone_lengths,
        "frames_used": len(chosen),
        "frames_available": len(usable),
        # False means too few frames cleared `geometry_from_confidence` and the
        # median was taken over every tracked frame instead, good ones and bad.
        "from_confident_frames": confident,
    }


# ---------------------------------------------------------------------------
# 4. Hold, extrapolate and decay
# ---------------------------------------------------------------------------


class FingerHold:
    """The cache-and-extrapolate state machine for one finger.

    While tracking is good it just records angles and angular velocity. When the
    gate says the finger has gone unreliable it stops taking new measurements
    and produces angles itself: dead reckoning from the cached velocity for a
    short window, then an exponential decay towards a relaxed pose.

    Decaying rather than extrapolating indefinitely is the important part.
    Unbounded dead reckoning is only plausible for about as long as a finger
    keeps doing what it was doing; past that it drifts into nonsense, and a
    relaxed hand is a far better guess than a linear extrapolation of one.
    """

    def __init__(self, config: dict, neutral: np.ndarray, fps: float):
        settings = config["hold"]
        self.fps = fps
        self.window = int(settings["velocity_window_frames"])
        self.max_velocity = float(settings["max_velocity_deg_per_second"])
        self.extrapolate_seconds = float(settings["extrapolate_seconds"])
        self.decay_seconds = max(float(settings["decay_seconds"]), 1e-6)
        self.neutral = neutral
        self.history: list = []
        self.cached: np.ndarray | None = None
        self.velocity = np.zeros(ANGLE_COUNT)
        self.frames_held = 0

    def observe(self, angles: np.ndarray) -> None:
        """Record a trusted frame, keeping just enough history to estimate velocity."""
        self.history.append(np.array(angles, dtype=float))
        if len(self.history) > self.window + 1:
            self.history.pop(0)
        self.cached = self.history[-1]
        self.frames_held = 0

    def _estimate_velocity(self) -> np.ndarray:
        """Degrees per second, averaged over the cached window.

        Averaged rather than taken from the last pair because a single noisy
        frame right before an occlusion would otherwise set the direction the
        finger travels for the whole hold. Clamped for the same reason: a
        velocity no finger could produce is a symptom of the bad tracking we are
        holding because of, not something to extrapolate.
        """
        if len(self.history) < 2:
            return np.zeros(ANGLE_COUNT)
        steps = np.diff(np.array(self.history), axis=0)
        velocity = steps.mean(axis=0) * self.fps
        magnitude = np.abs(velocity)
        excess = magnitude > self.max_velocity
        velocity[excess] = np.sign(velocity[excess]) * self.max_velocity
        return velocity

    def enter(self) -> None:
        """Freeze the cache and work out which way the finger was heading."""
        self.velocity = self._estimate_velocity()
        self.frames_held = 0

    def step(self) -> np.ndarray:
        """One held frame: dead reckoning inside the window, decay outside it."""
        self.frames_held += 1
        if self.cached is None:
            return np.array(self.neutral, dtype=float)
        elapsed = self.frames_held / self.fps
        reckoned = self.cached + self.velocity * min(elapsed, self.extrapolate_seconds)
        if elapsed <= self.extrapolate_seconds:
            return reckoned
        overrun = elapsed - self.extrapolate_seconds
        weight = 1.0 - math.exp(-overrun / self.decay_seconds)
        return (1.0 - weight) * reckoned + weight * self.neutral


def neutral_angles(config: dict) -> np.ndarray:
    """The relaxed pose a long hold decays into, as a 5x4 array."""
    pose = config["hold"]["neutral_pose"]
    rows = []
    for finger in FINGER_ORDER:
        if finger == "thumb":
            rows.append(
                [
                    pose["thumb_cmc_flexion"],
                    pose["thumb_cmc_abduction"],
                    pose["thumb_mcp_flexion"],
                    pose["thumb_ip_flexion"],
                ]
            )
        else:
            rows.append(
                [pose["mcp_flexion"], pose["mcp_abduction"], pose["pip_flexion"], pose["dip_flexion"]]
            )
    return np.array(rows, dtype=float)


# Fingers whose abduction values in stabilize.yaml - neutral pose and limits -
# are written in the palm frame, against the middle-finger axis, rather than
# against the clip's calibrated neutral. The four fingers' limits are clinical
# spread around each finger's own rest, which is what the calibration measures,
# so they apply to calibrated angles as they stand. The thumb's are explicitly
# not: the yaml describes them as the cone its metacarpal sweeps in the palm
# frame, so they have to be moved onto the calibrated zero before use.
PALM_FRAME_ABDUCTION = ("thumb",)


def calibrate_to_offsets(
    neutral: np.ndarray, limits: np.ndarray, offsets: np.ndarray | None
) -> tuple:
    """Put palm-frame neutral and limit values into the space angles are measured in.

    Measured abduction has the per-finger offset subtracted (see
    :func:`hand_to_angles`), so a palm-frame value must have it subtracted too.
    Left alone, a thumb resting at -30 degrees decayed through a long hold to
    -65, not the -35 the yaml asks for, and its limits were shifted by the same
    30 degrees.
    """
    neutral = np.array(neutral, dtype=float)
    limits = np.array(limits, dtype=float)
    if offsets is None:
        return neutral, limits
    for finger in PALM_FRAME_ABDUCTION:
        index = FINGER_ORDER.index(finger)
        neutral[index, 1] -= float(offsets[index])
        limits[index, 1, :] -= float(offsets[index])
    return neutral, limits


# ---------------------------------------------------------------------------
# 5. Anatomical joint limits
# ---------------------------------------------------------------------------


def limit_array(config: dict) -> np.ndarray:
    """The joint limits as a 5x4x2 array of (min, max) degrees.

    UNVERIFIED literature values - see the note in stabilize.yaml. They are an
    envelope for rejecting the impossible, not a measurement of anybody's hand.
    """
    limits = config["joint_limits"]
    return np.array(
        [[limits[finger][name] for name in ANGLE_NAMES] for finger in FINGER_ORDER], dtype=float
    )


def clamp_angles(angles: np.ndarray, limits: np.ndarray) -> tuple:
    """Hold every joint inside its range. Returns (clamped, how many were outside).

    Runs on every frame, not only during a hold, because most of the unnatural
    bending comes from the solver fitting noisy 2D data to impossible 3D while
    tracking is nominally fine - which no amount of occlusion handling catches.
    """
    clamped = np.clip(angles, limits[:, :, 0], limits[:, :, 1])
    violations = int(np.sum(np.abs(clamped - angles) > 1e-9))
    return clamped, violations


# ---------------------------------------------------------------------------
# 6. One Euro filter
# ---------------------------------------------------------------------------


class OneEuroFilter:
    """Adaptive low-pass, applied to the joint angles as the last pass.

    A fixed low-pass has to choose between jitter and lag. This one moves its
    cutoff with the signal's own speed: heavy smoothing when a joint is nearly
    still, which is where jitter is visible, and light smoothing when it moves
    fast, which is where lag is visible.

    Casey & Roussel's formulation, on an array of angles rather than a scalar.

    Running last means it cannot reintroduce an anatomically impossible angle,
    which would otherwise be a real worry about this ordering. Each output is
    ``alpha * value + (1 - alpha) * previous`` with alpha strictly between 0 and
    1 - a convex blend of two values that were already clamped, and by induction
    inside the same range. Verified on the sample clip: zero frames left the
    limits after filtering.
    """

    def __init__(self, config: dict, fps: float, shape: tuple):
        settings = config["one_euro"]
        self.enabled = bool(settings["enabled"])
        self.min_cutoff = float(settings["min_cutoff"])
        self.beta = float(settings["beta"])
        self.derivative_cutoff = float(settings["derivative_cutoff"])
        self.fps = fps
        self.previous: np.ndarray | None = None
        self.derivative = np.zeros(shape)

    def _alpha(self, cutoff):
        """Smoothing factor for a cutoff in Hz. Works on a scalar or an array."""
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau * max(self.fps, 1e-6))

    def __call__(self, value: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return value
        if self.previous is None:
            self.previous = np.array(value, dtype=float)
            return value
        rate = (value - self.previous) * self.fps
        alpha_derivative = self._alpha(self.derivative_cutoff)
        self.derivative = alpha_derivative * rate + (1.0 - alpha_derivative) * self.derivative
        # The cutoff rises with speed, so a fast joint is barely filtered and a
        # still one is filtered hard. This is the whole point of the filter.
        alpha = self._alpha(self.min_cutoff + self.beta * np.abs(self.derivative))
        smoothed = alpha * value + (1.0 - alpha) * self.previous
        self.previous = smoothed
        return smoothed


# ---------------------------------------------------------------------------
# The stage, assembled
# ---------------------------------------------------------------------------


def stabilize_hand(
    hands: list,
    scores: dict,
    label: str,
    fps: float,
    config: dict,
) -> tuple:
    """Run the whole pipeline over one hand's track. Returns (landmarks, debug).

    `hands` is one entry per frame, each a (21, 3) array or None. `scores` is
    that hand's slice of :func:`confidence_track`.
    """
    chirality = CHIRALITY[label]
    blend_frames = int(config["blend"]["frames"])
    smoothstep = config["blend"]["easing"] == "smoothstep"

    geometry = median_geometry(hands, scores["hand"], chirality, config)
    reconstruct = bool(config["reconstruct"]["enabled"]) and geometry is not None

    # Measured abduction reads against this hand's own neutral spread once
    # calibrated. The yaml's finger values already mean that; its thumb values
    # are palm-frame and are shifted onto the same zero here, or a held thumb
    # decays to - and is clamped around - the wrong spread.
    offsets = geometry["abduction_offsets"] if geometry else None
    neutral, limits = calibrate_to_offsets(neutral_angles(config), limit_array(config), offsets)

    holds = {finger: FingerHold(config, neutral[i], fps) for i, finger in enumerate(FINGER_ORDER)}
    gates = {finger: HysteresisGate(config) for finger in FINGER_ORDER}
    states = {finger: LIVE for finger in FINGER_ORDER}
    blend_left = {finger: 0 for finger in FINGER_ORDER}
    blend_from = {finger: None for finger in FINGER_ORDER}
    euro = OneEuroFilter(config, fps, (len(FINGER_ORDER), ANGLE_COUNT))

    output: list = []
    debug: list = []
    violations_total = 0

    for index, hand in enumerate(hands):
        measured = None
        basis = None
        if hand is not None and np.isfinite(hand).all():
            basis = palm_frame(hand)
            if basis is not None:
                measured = hand_to_angles(hand, chirality, offsets)

        angles = np.array(neutral, dtype=float)
        frame_states = {}
        for finger_index, finger in enumerate(FINGER_ORDER):
            score = scores["fingers"][index][finger_index]
            usable = measured is not None and np.isfinite(measured[finger_index]).all()
            reliable = gates[finger].update(score if usable else 0.0)
            hold = holds[finger]

            if reliable and usable:
                if states[finger] == HOLD:
                    # Tracking has come back. Do not snap - the real finger has
                    # usually moved while it was held, so ease across instead.
                    blend_left[finger] = blend_frames
                    blend_from[finger] = hold.step()
                    states[finger] = BLEND if blend_frames > 0 else LIVE
                if states[finger] == BLEND and blend_left[finger] > 0:
                    position = blend_frames - blend_left[finger] + 1
                    weight = position / blend_frames
                    if smoothstep:
                        weight = weight * weight * (3.0 - 2.0 * weight)
                    angles[finger_index] = (
                        1.0 - weight
                    ) * blend_from[finger] + weight * measured[finger_index]
                    blend_left[finger] -= 1
                    if blend_left[finger] == 0:
                        states[finger] = LIVE
                else:
                    states[finger] = LIVE
                    angles[finger_index] = measured[finger_index]
                hold.observe(measured[finger_index])
            else:
                if states[finger] != HOLD:
                    states[finger] = HOLD
                    hold.enter()
                angles[finger_index] = hold.step()
            frame_states[finger] = states[finger]

        angles, violations = clamp_angles(angles, limits)
        violations_total += violations
        # Smoothing last, and it cannot undo the clamp above: see OneEuroFilter.
        angles = euro(angles)

        if reconstruct and basis is not None:
            output.append(
                angles_to_hand(
                    angles,
                    geometry["palm_local"],
                    geometry["bone_lengths"],
                    basis,
                    chirality,
                    offsets,
                )
            )
        else:
            output.append(hand)

        debug.append(
            {
                "frame": index,
                "confidence": scores["hand"][index],
                "finger_confidence": scores["fingers"][index],
                "residual": scores["residual"][index],
                "palm_facing": scores["components"][index]["palm_facing"],
                "out_of_frame": scores["components"][index]["out_of_frame"],
                "states": dict(frame_states),
                "angles": angles.tolist(),
                "clamped": violations,
                "measured": scores["components"][index]["measured"],
            }
        )

    summary = {
        "reconstructed": reconstruct,
        "abduction_offsets_deg": (
            [round(float(v), 2) for v in geometry["abduction_offsets"]] if geometry else None
        ),
        "geometry_frames": geometry["frames_used"] if geometry else 0,
        "clamped_joint_frames": violations_total,
        "held_frames": sum(1 for row in debug if HOLD in row["states"].values()),
        "blend_frames": sum(1 for row in debug if BLEND in row["states"].values()),
    }
    return output, {"frames": debug, "summary": summary}
