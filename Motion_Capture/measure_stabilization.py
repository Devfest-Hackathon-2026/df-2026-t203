"""Before-and-after statistics for stage 2.5, as a regression check.

    python measure_stabilization.py --raw output/Water/landmarks.json \\
        --filtered output/Water/landmarks_filtered.json \\
        --stabilized output/Water/landmarks_stabilized.json

Compares the same frames at three points in the pipeline and prints four
measures. Read them in this order, because they are not equally meaningful:

**Joint-limit violations** - how many of the 20 joint angles fall outside the
anatomical range. This is the honest headline: it counts poses no hand can make,
it is measured the same way on both sides, and nothing about the reconstruction
forces it to improve.

**Angle jitter** - mean absolute second difference of each joint angle, i.e. how
much the angular velocity changes frame to frame. Responds to shaking rather
than to how fast the sign is.

**Angle step** - mean absolute frame-to-frame change. Falling too far is a
warning, not a win: it would mean real motion is being smoothed away.

**Rigidity residual** - the spread of bone lengths against their median. Useful
as a *before* number, because it is what detects bad frames in the first place.
Nearly useless as an *after* number: stage 2.5 rebuilds the hand by forward
kinematics on fixed bone lengths, so the residual is ~0 by construction whatever
the input was. It is reported because it was asked for and because the before
column is real, not because a small after number proves anything.

Only frames where the hand was genuinely detected are compared, so the numbers
are not diluted by frames that were filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import hand_stabilizer
from hand_stabilizer import CHIRALITY, HAND_LABELS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Before/after statistics for the hand stabiliser.")
    parser.add_argument("--raw", required=True, type=Path, help="landmarks.json from stage 1.")
    parser.add_argument("--filtered", type=Path, default=None, help="landmarks_filtered.json from stage 2.")
    parser.add_argument(
        "--stabilized", required=True, type=Path, help="landmarks_stabilized.json from stage 2.5."
    )
    parser.add_argument("--config", type=Path, default=None, help="stabilize.yaml to read limits from.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help=(
            "Degrees of overshoot to ignore when counting joint-limit violations. Not slack in "
            "the limits: landmark coordinates are stored as metres rounded to 5 decimals, so "
            "reading angles back out of a JSON file costs up to ~0.2 degrees on the shortest "
            "bones. 0 counts every violation including that quantisation."
        ),
    )
    return parser.parse_args(argv)


def load(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"Not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def frame_offset(data: dict) -> int:
    """How many frames stage 2 trimmed off the front, so indices line up."""
    return int(((data.get("post_processing") or {}).get("trim") or {}).get("dropped_before", 0))


def hand_series(data: dict, label: str, offset: int = 0) -> dict:
    """Raw-frame-index -> (21, 3) array, for every frame carrying this hand."""
    series = {}
    for index, frame in enumerate(data.get("frames", [])):
        points = (frame.get("hands") or {}).get(label)
        if points is not None:
            series[index + offset] = np.asarray(points, dtype=float)
    return series


def measured_frames(raw: dict, label: str) -> set:
    """Frames where MediaPipe actually detected this hand, before any filling."""
    return {
        index
        for index, frame in enumerate(raw.get("frames", []))
        if label in (frame.get("hands") or {})
    }


def statistics(
    series: dict,
    frames: list,
    label: str,
    limits: np.ndarray,
    tolerance: float = 0.0,
    offsets: np.ndarray | None = None,
) -> dict:
    """Residual, limit-violation and jitter statistics over the given frames.

    `offsets` are the abduction calibration stage 2.5 used. Every column is
    measured with the same ones, so the before and after numbers share a zero -
    re-deriving them per column would compare two different definitions of
    "no spread" and quietly flatter the output.
    """
    chirality = CHIRALITY[label]
    hands = [series[index] for index in frames if index in series]
    if len(hands) < 3:
        return {}

    lengths = np.array([hand_stabilizer.segment_lengths(hand) for hand in hands])
    reference = np.median(lengths, axis=0)
    residuals = np.array([hand_stabilizer.rigidity_residual(row, reference) for row in lengths])

    angles, keys = [], []
    for index in frames:
        if index not in series:
            continue
        measured = hand_stabilizer.hand_to_angles(series[index], chirality, offsets)
        if measured is not None and np.isfinite(measured).all():
            angles.append(measured)
            keys.append(index)
    if len(angles) < 3:
        return {}
    angles = np.array(angles)

    overshoot = np.maximum(limits[:, :, 0] - angles, angles - limits[:, :, 1]).clip(min=0.0)
    outside = overshoot > tolerance
    # Only difference across frames that are actually adjacent, so a gap in the
    # series is not counted as a huge single-frame jump.
    adjacent = np.array([k + 1 == n for k, n in zip(keys, keys[1:])])
    steps = np.abs(np.diff(angles, axis=0))[adjacent]
    consecutive = np.array(
        [a and b for a, b in zip(adjacent, list(adjacent[1:]) + [False])]
    )
    jitter = np.abs(np.diff(angles, n=2, axis=0))[consecutive[: len(angles) - 2]]

    return {
        "frames": len(angles),
        "residual_median": float(np.median(residuals)),
        "residual_p90": float(np.percentile(residuals, 90)),
        "residual_max": float(np.max(residuals)),
        "violation_rate": float(outside.mean()),
        "violation_frames": float(outside.any(axis=(1, 2)).mean()),
        "worst_violation_deg": float(overshoot.max()),
        "step": float(steps.mean()) if steps.size else float("nan"),
        "jitter": float(jitter.mean()) if jitter.size else float("nan"),
    }


def report(label: str, columns: dict) -> None:
    names = [name for name, values in columns.items() if values]
    if not names:
        print(f"\n{label} hand: not enough data to compare")
        return

    print(f"\n{label} hand - {columns[names[0]]['frames']} frames compared")
    rows = [
        ("joint-limit violations, per angle", "violation_rate", "{:.1%}"),
        ("  frames with any violation", "violation_frames", "{:.1%}"),
        ("  worst overshoot", "worst_violation_deg", "{:.1f} deg"),
        ("angle jitter (mean |2nd diff|)", "jitter", "{:.2f} deg"),
        ("angle step (mean |1st diff|)", "step", "{:.2f} deg"),
        ("rigidity residual, median", "residual_median", "{:.1%}"),
        ("  p90", "residual_p90", "{:.1%}"),
        ("  max", "residual_max", "{:.1%}"),
    ]
    width = max(len(text) for text, _, _ in rows)
    print("  " + " " * width + "".join(f"{name:>18}" for name in names))
    for text, key, fmt in rows:
        cells = "".join(f"{fmt.format(columns[name][key]):>18}" for name in names)
        print(f"  {text:<{width}}{cells}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load(args.raw)
    stabilized = load(args.stabilized)
    filtered = load(args.filtered) if args.filtered else None
    config = hand_stabilizer.load_config(args.config)
    limits = hand_stabilizer.limit_array(config)

    offset = frame_offset(stabilized)
    print(f"source: {args.raw.name} -> {args.stabilized.name}")
    print(f"stage 2 trimmed {offset} frames off the front; indices are aligned to the raw clip")

    for label in HAND_LABELS:
        detected = measured_frames(raw, label)
        series = {
            "raw": hand_series(raw, label),
            "stabilised": hand_series(stabilized, label, offset),
        }
        if filtered is not None:
            series["stage 2 only"] = hand_series(filtered, label, frame_offset(filtered))
        # Compare only frames every column actually has, and that were genuinely
        # detected rather than filled in.
        shared = sorted(detected.intersection(*[set(s) for s in series.values()]))
        recorded = ((stabilized.get("stabilization") or {}).get("per_hand") or {}).get(label) or {}
        offsets = recorded.get("abduction_offsets_deg")
        offsets = np.asarray(offsets, dtype=float) if offsets else None
        columns = {
            name: statistics(points, shared, label, limits, args.tolerance, offsets)
            for name, points in (
                ("raw", series["raw"]),
                *( [("stage 2 only", series["stage 2 only"])] if filtered is not None else [] ),
                ("stabilised", series["stabilised"]),
            )
        }
        report(label, columns)

    print(
        "\nNote: the rigidity residual of the stabilised column is ~0 by construction - the hand "
        "\nis rebuilt by forward kinematics on fixed bone lengths, so it cannot be otherwise. "
        "\nThe joint-limit and jitter rows are the ones that carry information."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
