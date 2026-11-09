"""Stage 1 of the single-video pipeline: video -> landmark JSON.

Runs MediaPipe on one video file and writes every frame's landmarks to a single
JSON file. Nothing here touches Blender: MediaPipe cannot be imported from
Blender's bundled Python, so extraction and retargeting are deliberately two
separate processes that only share a file.

What gets written per frame:
  * ``pose_world`` - MediaPipe Pose *world* landmarks (33 points, metres, origin
    at the midpoint of the hips). Used for the torso, head and arms.
  * ``hands``      - MediaPipe Hands *world* landmarks (21 points per hand,
    metres, origin at the wrist). Used for finger shape and palm orientation.

Both are metric and origin-relative, which is what makes a monocular retarget
possible at all: we only ever use directions and hand-local shape, never an
absolute position in the room, so no camera calibration is needed.

A hand is only recorded when it is genuinely in shot. MediaPipe reports all 21
landmarks for a hand that is half outside the picture, extrapolating the ones it
cannot see, and on the sample clip the signer's resting hand sits below the
bottom edge for the whole take - detected every frame, 57% of it invented.
``--max-out-of-frame`` throws those away, so a hand nobody could see is left
unposed rather than posed from a guess.

Not solved here, and not solvable from one uncalibrated camera: where the person
is standing, how far they are from the camera, and true global depth. The
retarget stage therefore keeps the avatar's root planted.

Run with a Python that has mediapipe installed - in this repo that is
``.venv311``, not the default ``.venv`` (MediaPipe 0.10.9 has no wheel for
Python 3.14):

    .venv311/Scripts/python.exe extract_landmarks.py \
        --video src/avatar/videos/Water.mp4 \
        --out   src/avatar/new-pipeline-mocap/output/Water/landmarks.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import hand_stabilizer

POSE_LANDMARK_NAMES = [landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose and hand world landmarks from a single video.",
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video file (one video, one person).")
    parser.add_argument("--out", required=True, type=Path, help="Where to write landmarks.json.")
    parser.add_argument(
        "--selfie",
        action="store_true",
        help=(
            "Set this when the video is already mirrored (a webcam selfie recording). "
            "Only affects the fallback path: hands are normally assigned by matching them "
            "to the pose's wrists, and MediaPipe's handedness - which assumes a mirrored "
            "image - is consulted only when those wrists are not visible."
        ),
    )
    parser.add_argument(
        "--pose-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe Pose model complexity (2 = most accurate, slowest).",
    )
    parser.add_argument("--hand-complexity", type=int, default=1, choices=[0, 1], help="MediaPipe Hands model complexity.")
    parser.add_argument(
        "--max-out-of-frame",
        type=float,
        default=0.3,
        help=(
            "Reject a detected hand when more than this fraction of its 21 landmarks fall "
            "outside the picture, because MediaPipe extrapolates those rather than seeing them. "
            "1.0 keeps every detection."
        ),
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.4,
        help=(
            "Exponential smoothing factor in [0, 1). 0 disables it. Higher means smoother "
            "and laggier - finger detail is the first thing to go, so keep it low for signing."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = whole video). Useful for a quick test.",
    )
    parser.add_argument(
        "--debug-video",
        type=Path,
        default=None,
        help="Optional path for an annotated copy of the video, to eyeball what MediaPipe saw.",
    )
    return parser.parse_args(argv)


LEFT_WRIST = 15
RIGHT_WRIST = 16


def out_of_frame_fraction(hand_landmarks) -> float:
    """How much of a detected hand lies outside the picture.

    MediaPipe still reports all 21 landmarks for a hand that is half out of
    shot, extrapolating the ones it cannot see. Those are guesses, and they look
    like a real handshape to everything downstream.
    """
    landmarks = hand_landmarks.landmark
    outside = sum(1 for lm in landmarks if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0))
    return outside / len(landmarks)


def assign_hands(
    hands_result, pose_result, width: int, height: int, selfie: bool, max_out_of_frame: float = 1.0
) -> tuple:
    """Work out which detected hand is the left one and which is the right.

    MediaPipe's own handedness classifier is unreliable exactly where it matters:
    when the signer's hands meet, it flips on individual frames (scores around
    0.55) and the retarget swaps the two handshapes for a frame or two, which
    reads as a violent snap. Nothing downstream can undo that - a filter just
    smears the wrong hand across its neighbours.

    The pose model, however, tracks both wrists steadily. So each detected hand
    is matched to the nearer pose wrist in image space, choosing the pairing with
    the lower total distance. On the sample clip the winning distances are a few
    pixels against a couple of hundred for the alternative, so the assignment is
    not a close call. The classifier is only consulted when the pose wrists are
    unavailable.

    Detections that are mostly outside the picture are thrown away first: a hand
    resting below the bottom of frame is still "detected", but its landmarks are
    extrapolation, and posing the avatar from them looks worse than not posing
    that hand at all.

    Returns (hands by label, per-hand confidence signals, number of detections
    placed by the classifier, number where the two methods disagreed, number
    dropped as out of frame).
    """
    detections = [
        detection
        for detection in zip(
            hands_result.multi_hand_landmarks or [],
            hands_result.multi_hand_world_landmarks or [],
            hands_result.multi_handedness or [],
        )
        if out_of_frame_fraction(detection[0]) <= max_out_of_frame
    ]
    dropped = len(hands_result.multi_hand_landmarks or []) - len(detections)
    if not detections:
        return {}, {}, 0, 0, dropped

    def classifier_label(handedness) -> str:
        label = handedness.classification[0].label
        # MediaPipe assumes a mirrored (selfie) image, so on ordinary footage its
        # "Left" is really the person's right hand.
        return label if selfie else ("Right" if label == "Left" else "Left")

    pose_wrists = {}
    if pose_result.pose_landmarks is not None:
        for label, index in (("Left", LEFT_WRIST), ("Right", RIGHT_WRIST)):
            landmark = pose_result.pose_landmarks.landmark[index]
            pose_wrists[label] = (landmark.x * width, landmark.y * height)

    def distance(detection, label: str) -> float:
        wrist = detection[0].landmark[0]
        target = pose_wrists[label]
        return math.hypot(wrist.x * width - target[0], wrist.y * height - target[1])

    def match_limit(detection) -> float:
        """How far a pose wrist may sit from this hand's own wrist and still be it.

        Scaled by the hand's size on screen rather than a fixed pixel count, so it
        travels between resolutions and between a hand near the camera and one far
        from it. A correct match lands within a few pixels; the wrong wrist is
        typically more than a hand-length away. MediaPipe's pose visibility score
        is deliberately not used as a gate - it drops below 0.5 on a third of the
        frames of the sample clip while the wrist position stays perfectly usable.
        """
        wrist = detection[0].landmark[0]
        span = max(
            math.hypot((landmark.x - wrist.x) * width, (landmark.y - wrist.y) * height)
            for landmark in detection[0].landmark
        )
        return max(span, 0.05 * math.hypot(width, height))

    assignment: dict = {}  # label -> index into detections
    if len(detections) == 2 and len(pose_wrists) == 2:
        straight = distance(detections[0], "Left") + distance(detections[1], "Right")
        crossed = distance(detections[0], "Right") + distance(detections[1], "Left")
        order = ("Left", "Right") if straight <= crossed else ("Right", "Left")
        assignment = {order[0]: 0, order[1]: 1}
    elif pose_wrists:
        for index, detection in enumerate(detections):
            label = min(pose_wrists, key=lambda candidate: distance(detection, candidate))
            rival = assignment.get(label)
            if rival is None or distance(detection, label) < distance(detections[rival], label):
                assignment[label] = index

    # Drop any match that is too far from the wrist it was paired with; the
    # classifier gets a turn at those below.
    assignment = {
        label: index
        for label, index in assignment.items()
        if distance(detections[index], label) <= match_limit(detections[index])
    }

    from_classifier = 0
    for index, detection in enumerate(detections):  # whatever geometry could not place
        if index in assignment.values():
            continue
        label = classifier_label(detection[2])
        if label not in assignment:
            assignment[label] = index
            from_classifier += 1

    disagreements = sum(
        1 for label, index in assignment.items() if classifier_label(detections[index][2]) != label
    )
    hands = {label: landmark_list_to_xyz(detections[index][1]) for label, index in assignment.items()}
    confidence = {
        label: measure_confidence(detections[index]) for label, index in assignment.items()
    }
    return hands, confidence, from_classifier, disagreements, dropped


def measure_confidence(detection) -> dict:
    """The raw signals stage 2.5 turns into a confidence score for this hand.

    MediaPipe's hand solution reports ``visibility`` and ``presence`` as exactly
    0.0 on every landmark - image-space and world, verified on this repo's
    sample footage - so there is no tracking confidence to read. What follows is
    everything measurable that stands in for one, recorded here and combined
    downstream, where the weights live in ``stabilize.yaml``.

    Two of these cannot be recovered later, which is why they are written now
    rather than derived in a later stage: ``out_of_frame`` needs the image-space
    landmarks, which are not saved, and ``handedness_score`` comes from a
    classifier output that is discarded once the sides are assigned. The other
    two are geometric, and are measured here so that they describe what the
    tracker actually reported - by stage 2.5 the Butterworth filter has already
    made the geometry look more plausible than it was.
    """
    image_landmarks, world_landmarks, handedness = detection
    world = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark])
    return {
        "out_of_frame": round(out_of_frame_fraction(image_landmarks), 4),
        "handedness_score": round(float(handedness.classification[0].score), 4),
        "palm_facing": round(hand_stabilizer.palm_facing(world), 4),
        # Bone lengths do not change on a real hand, so the spread of these
        # against their own median is the strongest evidence available that a
        # frame was solved badly.
        "segment_lengths": [round(value, 5) for value in hand_stabilizer.segment_lengths(world)],
    }


def landmark_list_to_xyz(landmark_list, with_visibility: bool = False) -> list[list[float]]:
    if with_visibility:
        return [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmark_list.landmark]
    return [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]


def smooth_track(track: list, alpha: float) -> None:
    """Exponentially smooth one landmark track in place, resetting across gaps.

    ``track`` holds one entry per frame: either a list of [x, y, z(, visibility)]
    points, or None when the detector lost the subject. Smoothing restarts after
    every gap so a re-detection is not dragged towards where the hand used to be.
    """
    if alpha <= 0.0:
        return
    previous = None
    for frame_index, points in enumerate(track):
        if points is None:
            previous = None
            continue
        if previous is None:
            previous = [list(point) for point in points]
            continue
        smoothed = [
            [alpha * previous[i][axis] + (1.0 - alpha) * points[i][axis] for axis in range(len(points[i]))]
            for i in range(len(points))
        ]
        track[frame_index] = smoothed
        previous = smoothed


def round_points(points, digits: int = 5):
    if points is None:
        return None
    return [[round(value, digits) for value in point] for point in points]


def extract(args: argparse.Namespace) -> dict:
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open the video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[extract] {video_path.name}: {width}x{height} @ {fps:.2f} fps, ~{reported_frames} frames", flush=True)

    writer = None
    drawing = None
    if args.debug_video is not None:
        args.debug_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.debug_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        drawing = mp.solutions.drawing_utils

    pose_track: list = []
    confidence_track: list = []
    classifier_fallbacks = 0
    classifier_disagreements = 0
    out_of_frame_drops = 0
    hand_tracks: dict = {"Left": [], "Right": []}

    pose_solution = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.pose_complexity,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    hands_solution = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=args.hand_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    frame_index = 0
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            pose_result = pose_solution.process(rgb_frame)
            hands_result = hands_solution.process(rgb_frame)

            if pose_result.pose_world_landmarks is not None:
                pose_track.append(landmark_list_to_xyz(pose_result.pose_world_landmarks, with_visibility=True))
            else:
                pose_track.append(None)

            frame_hands, frame_confidence, from_classifier, disagreements, dropped = assign_hands(
                hands_result, pose_result, width, height, args.selfie, args.max_out_of_frame
            )
            classifier_fallbacks += from_classifier
            classifier_disagreements += disagreements
            out_of_frame_drops += dropped

            confidence_track.append(frame_confidence)
            for label in ("Left", "Right"):
                hand_tracks[label].append(frame_hands.get(label))

            if writer is not None:
                annotated = bgr_frame.copy()
                if pose_result.pose_landmarks is not None:
                    drawing.draw_landmarks(annotated, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                for hand_landmarks in hands_result.multi_hand_landmarks or []:
                    drawing.draw_landmarks(annotated, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                writer.write(annotated)

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"[extract] frame {frame_index}", flush=True)
    finally:
        capture.release()
        pose_solution.close()
        hands_solution.close()
        if writer is not None:
            writer.release()

    smooth_track(pose_track, args.smoothing)
    for label in ("Left", "Right"):
        smooth_track(hand_tracks[label], args.smoothing)

    frames = []
    for index in range(frame_index):
        hands_this_frame = {
            label: hand_tracks[label][index]
            for label in ("Left", "Right")
            if hand_tracks[label][index] is not None
        }
        frames.append(
            {
                "frame": index,
                "pose_world": round_points(pose_track[index]),
                "hands": {label: round_points(points) for label, points in hands_this_frame.items()},
                # Measured on the raw detection, before any smoothing, and only
                # for hands that were genuinely detected on this frame. A label
                # missing from here means the tracker did not see that hand,
                # which downstream reads as no confidence at all.
                "hand_confidence": {
                    label: signals
                    for label, signals in confidence_track[index].items()
                    if hand_tracks[label][index] is not None
                },
            }
        )

    pose_found = sum(1 for frame in frames if frame["pose_world"] is not None)
    left_found = sum(1 for frame in frames if "Left" in frame["hands"])
    right_found = sum(1 for frame in frames if "Right" in frame["hands"])
    print(
        f"[extract] {frame_index} frames processed - pose on {pose_found}, "
        f"left hand on {left_found}, right hand on {right_found}",
        flush=True,
    )
    print(
        f"[extract] hand sides taken from the pose wrists; MediaPipe's own handedness would "
        f"have disagreed on {classifier_disagreements} detections, and was used as a fallback "
        f"for {classifier_fallbacks}",
        flush=True,
    )
    if out_of_frame_drops:
        print(
            f"[extract] rejected {out_of_frame_drops} hand detections that were more than "
            f"{args.max_out_of_frame:.0%} outside the picture",
            flush=True,
        )

    return {
        "schema": "sign-language-ai/new-pipeline-mocap/landmarks/2",
        "source_video": video_path.name,
        "fps": fps,
        "frame_count": frame_index,
        "image_size": [width, height],
        "selfie_input": bool(args.selfie),
        "max_out_of_frame": args.max_out_of_frame,
        "smoothing": args.smoothing,
        "coordinate_notes": {
            "pose_world": (
                "MediaPipe pose world landmarks: metres, origin at hip midpoint, "
                "+x image-right, +y image-down, +z away from camera."
            ),
            "hands": (
                "MediaPipe hand world landmarks: metres, origin at that hand's wrist, "
                "same axis convention."
            ),
            "hand_confidence": (
                "Raw signals for stage 2.5, measured on the unsmoothed detection. MediaPipe "
                "reports visibility and presence as 0.0 on every hand landmark, so these "
                "stand in for the confidence score it does not provide: out_of_frame is the "
                "fraction of the 21 landmarks outside the picture, handedness_score is the "
                "classifier's own certainty, palm_facing is |palm normal . camera axis| "
                "(1.0 square to the camera, 0.0 edge-on), and segment_lengths are the 20 "
                "bone lengths in metres, four per finger in thumb-to-pinky order."
            ),
        },
        "pose_landmark_names": POSE_LANDMARK_NAMES,
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = extract(args)
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(f"[extract] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
