"""Stage 2.5 of the single-video pipeline: stabilise the hands.

Sits between freemocap's smoothing and the Blender retarget, and exists because
of one failure the earlier stages cannot address: when a hand turns side-on to
the camera, or a finger goes behind another, MediaPipe keeps reporting all 21
landmarks with no signal that it is guessing, and the fingers snap into bends no
hand makes.

    landmarks_filtered.json -> stabilize_hands.py -> landmarks_stabilized.json

The output is the same schema as the input, so stage 3 does not know this stage
exists and ``--no-stabilize`` skips it entirely.

What it does, per finger, per frame::

    confidence -> hysteresis -> hold/extrapolate/decay -> blend back
               -> anatomical clamp -> one euro -> forward kinematics

The mechanics live in ``hand_stabilizer.py`` and every threshold in
``stabilize.yaml``. This file is the plumbing: read, run, report, write.

Two things worth knowing before reading the output:

* **The hands are rebuilt, not adjusted.** Landmarks come out of forward
  kinematics from clamped joint angles on median bone lengths, so bone length
  and palm size are exactly constant by construction. That removes a large part
  of what "anatomically impossible" looked like - but it also means the rigidity
  residual of the output is ~0 whatever the input was, so that number is not
  evidence of anything. Judge it on the angle statistics and the debug overlay.
* **Held frames are invented.** A finger under hold is not being measured. The
  debug JSON records the state of every finger on every frame, and
  ``hands_reliable`` in the output says which hands were trusted at all.

Run standalone with any Python that has numpy and PyYAML:

    python stabilize_hands.py --landmarks landmarks_filtered.json \\
        --out landmarks_stabilized.json --debug-out stabilize_debug.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import hand_stabilizer
from hand_stabilizer import FINGER_ORDER, HAND_LABELS, HOLD


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Occlusion handling, joint limits and adaptive smoothing for tracked hands.",
    )
    parser.add_argument("--landmarks", required=True, type=Path, help="Landmarks from stage 1 or 2.")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the stabilised landmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="stabilize.yaml to read (default: the one next to this script). Every tunable is in it.",
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=None,
        help=(
            "Also write a per-frame trace: confidence, hold state, rigidity residual and joint "
            "angles. This is what debug_overlay.py draws."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the sampling rate. By default the frame rate recorded in the input is used.",
    )
    return parser.parse_args(argv)


def hand_track(frames: list, label: str) -> list:
    """One hand's landmarks as an array per frame, or None where it is absent."""
    track = []
    for frame in frames:
        points = (frame.get("hands") or {}).get(label)
        track.append(np.asarray(points, dtype=float) if points is not None else None)
    return track


def angle_statistics(debug: dict) -> dict:
    """Jitter and joint-limit pressure, the two numbers worth watching.

    Jitter is the mean absolute second difference of each joint angle - how much
    the angular *velocity* changes between frames. It is the measure that
    responds to the shaking this stage exists to remove, where a plain
    frame-to-frame difference mostly measures how fast the sign is.
    """
    angles = np.array([row["angles"] for row in debug["frames"]])
    if angles.shape[0] < 3:
        return {}
    return {
        "mean_step_deg": round(float(np.mean(np.abs(np.diff(angles, axis=0)))), 3),
        "jitter_deg": round(float(np.mean(np.abs(np.diff(angles, n=2, axis=0)))), 3),
        "worst_step_deg": round(float(np.max(np.abs(np.diff(angles, axis=0)))), 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    landmarks_path = args.landmarks.expanduser().resolve()
    if not landmarks_path.is_file():
        raise SystemExit(f"Landmark file not found: {landmarks_path}")

    data_in = json.loads(landmarks_path.read_text(encoding="utf-8"))
    frames = data_in.get("frames")
    if not frames:
        raise SystemExit(f"{landmarks_path} contains no frames.")

    config = hand_stabilizer.load_config(args.config)
    fps = args.fps or data_in.get("fps") or 30.0

    if not any(frame.get("hand_confidence") for frame in frames):
        print(
            "[stabilize] no hand_confidence block in the input - this landmarks.json predates "
            "stage 1's confidence signals. Falling back to geometry measured here, which is "
            "weaker: by this point the smoothing has already tidied it up.",
            flush=True,
        )

    scores = hand_stabilizer.confidence_track(frames, config)
    summaries = {}
    debug_out = {}

    for label in HAND_LABELS:
        track = hand_track(frames, label)
        present = sum(1 for hand in track if hand is not None)
        if present == 0:
            print(f"[stabilize] {label}: never tracked, nothing to do")
            continue

        stabilized, debug = hand_stabilizer.stabilize_hand(track, scores[label], label, fps, config)

        for index, hand in enumerate(stabilized):
            if hand is None:
                frames[index].get("hands", {}).pop(label, None)
                continue
            frames[index].setdefault("hands", {})[label] = [
                [round(float(value), 5) for value in point] for point in hand
            ]

        summary = debug["summary"]
        summary.update(angle_statistics(debug))
        # Frames where at least one finger was being held rather than measured.
        held_fingers = sum(
            sum(1 for state in row["states"].values() if state == HOLD) for row in debug["frames"]
        )
        summary["held_finger_frames"] = held_fingers
        summary["finger_frames"] = len(debug["frames"]) * len(FINGER_ORDER)
        summaries[label] = summary
        debug_out[label] = debug

        print(
            f"[stabilize] {label}: {present}/{len(frames)} frames tracked, "
            f"{held_fingers}/{summary['finger_frames']} finger-frames held "
            f"({held_fingers / max(summary['finger_frames'], 1):.0%}), "
            f"{summary['clamped_joint_frames']} joint angles clamped, "
            f"jitter {summary.get('jitter_deg', float('nan')):.2f} deg",
            flush=True,
        )
        if not summary["reconstructed"]:
            # Say which: an operator who switched reconstruction off should not
            # be told their footage was too poor, and vice versa.
            reason = (
                "not enough usable geometry to rebuild the hand"
                if config["reconstruct"]["enabled"]
                else "reconstruction is disabled in the config (reconstruct.enabled: false)"
            )
            print(
                f"[stabilize] {label}: {reason} - "
                "the joint limits could not be applied and the landmarks are unchanged",
                flush=True,
            )

    data_out = dict(data_in)
    data_out["frames"] = frames
    data_out["stabilization"] = {
        "config": str((args.config or hand_stabilizer.CONFIG_PATH).name),
        "fps": fps,
        "per_hand": summaries,
        "note": (
            "Fingers under 'hold' were extrapolated or decayed towards a relaxed pose, not "
            "measured. Landmarks are rebuilt by forward kinematics from clamped joint angles "
            "on median bone lengths, so bone lengths are constant by construction and the "
            "rigidity residual of this output is not evidence of tracking quality. Joint "
            "limits are unverified literature values - see stabilize.yaml."
        ),
    }

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data_out), encoding="utf-8")
    print(f"[stabilize] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.debug_out is not None:
        debug_path = args.debug_out.expanduser().resolve()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps(
                {
                    "source": landmarks_path.name,
                    "fps": fps,
                    "fingers": list(FINGER_ORDER),
                    "threshold": config["confidence"]["threshold"],
                    "abduction_offsets": {
                        label: summaries[label].get("abduction_offsets_deg")
                        for label in debug_out
                    },
                    "trim": (data_in.get("post_processing") or {}).get("trim", {}),
                    "hands": debug_out,
                }
            ),
            encoding="utf-8",
        )
        print(f"[stabilize] wrote {debug_path} ({debug_path.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
