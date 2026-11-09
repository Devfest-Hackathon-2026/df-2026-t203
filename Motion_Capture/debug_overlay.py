"""Render what the stabiliser decided, frame-synced to the source video.

    python debug_overlay.py --video Water.mp4 --debug stabilize_debug.json \\
        --before landmarks.json --after landmarks_stabilized.json --out debug_hands.mp4

The point of this is to be able to *look* at a specific frame rather than trust a
summary statistic. Aggregate numbers cannot tell you whether the frames where a
palm turns edge-on are the ones being caught - they can only tell you that
something was caught. So every panel here is per-frame and lines up with the
video:

  * the source frame, so you can see what the hand was actually doing;
  * per hand, the raw and stabilised handshape drawn side by side, projected
    onto the camera plane - a finger snapping into an impossible bend is obvious
    here and invisible in any average;
  * per finger, its confidence and whether it is live, held or blending back;
  * a scrolling chart of hand confidence and rigidity residual against the
    threshold, with the hold and blend runs shaded underneath, so you can see a
    hold coming before it happens.

Needs numpy and cv2. The landmark files are optional: without them you get the
confidence and state panels but no wireframes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# BGR, because cv2. Chosen to stay distinguishable in greyscale too.
COLOURS = {
    "live": (90, 200, 90),
    "hold": (60, 170, 240),
    "blend": (235, 175, 95),
    "raw": (140, 140, 245),
    "stable": (150, 230, 150),
    "text": (235, 235, 235),
    "dim": (130, 130, 130),
    "panel": (32, 32, 34),
    "grid": (62, 62, 66),
    "threshold": (90, 90, 235),
    "residual": (220, 200, 90),
}

# The 21-landmark hand skeleton, as pairs to draw.
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),  # the knuckle line, so the palm reads as a palm
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SIZE = 512
PANEL_WIDTH = 700
CHART_HEIGHT = 150


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the hand stabiliser's per-frame decisions.")
    parser.add_argument("--video", required=True, type=Path, help="The source video the capture came from.")
    parser.add_argument("--debug", required=True, type=Path, help="stabilize_debug.json from stage 2.5.")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the overlay video.")
    parser.add_argument(
        "--before",
        type=Path,
        default=None,
        help="Raw landmarks.json from stage 1, drawn as the 'raw' wireframe.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=None,
        help="landmarks_stabilized.json from stage 2.5, drawn as the 'stabilised' wireframe.",
    )
    parser.add_argument(
        "--chart-seconds",
        type=float,
        default=4.0,
        help="How much history the scrolling charts show.",
    )
    return parser.parse_args(argv)


def load_hands(path: Path | None) -> dict:
    """Per-label, per-frame landmark arrays from a landmarks file.

    Leaving the file out is allowed, but naming one that does not exist is an
    error: returning nothing would draw every wireframe as "not tracked", which
    looks exactly like a tracking failure and sends you debugging the wrong thing.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"Landmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks: dict = {}
    for index, frame in enumerate(data.get("frames", [])):
        for label, points in (frame.get("hands") or {}).items():
            tracks.setdefault(label, {})[index] = np.asarray(points, dtype=float)
    return tracks


def draw_hand(canvas, origin, size, hand, colour, title, subtitle="") -> None:
    """Draw one handshape, orthographically, as the camera saw it.

    Wrist-relative metres projected straight onto the image plane: x right,
    y down, depth thrown away. Depth is the axis a single camera is worst at, so
    leaving it out shows the handshape the tracker actually committed to.
    """
    x0, y0 = origin
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), COLOURS["panel"], -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), COLOURS["grid"], 1)
    cv2.putText(canvas, title, (x0 + 6, y0 + 16), FONT, 0.38, COLOURS["dim"], 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (x0 + 6, y0 + size - 8), FONT, 0.36, colour, 1, cv2.LINE_AA)

    if hand is None or not np.isfinite(hand).all():
        cv2.putText(
            canvas, "not tracked", (x0 + size // 2 - 32, y0 + size // 2),
            FONT, 0.38, COLOURS["dim"], 1, cv2.LINE_AA,
        )
        return

    # A fixed scale, so a hand that collapses towards its wrist looks collapsed
    # rather than being silently zoomed back up to fill the box. `span` is the
    # half-width in metres that fills the panel - a landmark 0.13 m from the
    # wrist reaches the edge, which keeps a splayed hand inside the box.
    span = 0.13
    centre = np.array([x0 + size / 2, y0 + size / 2])
    points = centre + (hand[:, :2] - hand[0, :2]) / span * (size * 0.45)
    # Anything still outside is pinned to the border rather than drawn over the
    # neighbouring panel.
    points = np.clip(
        points, [x0 + 2, y0 + 2], [x0 + size - 2, y0 + size - 2]
    ).astype(int)

    for first, second in BONES:
        cv2.line(canvas, tuple(points[first]), tuple(points[second]), colour, 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(canvas, tuple(point), 3 if index in (0, 4, 8, 12, 16, 20) else 2, colour, -1, cv2.LINE_AA)


def draw_hand_panel(canvas, origin, width, height, label, record, raw_hand, stable_hand) -> None:
    """One hand's block: numbers, per-finger states, and the two wireframes."""
    x0, y0 = origin
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), COLOURS["panel"], -1)

    measured = record["measured"] if record else False
    cv2.putText(canvas, f"{label} hand", (x0 + 10, y0 + 22), FONT, 0.55, COLOURS["text"], 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "detected" if measured else "not measured this frame",
        (x0 + 105, y0 + 22),
        FONT, 0.38,
        COLOURS["live"] if measured else COLOURS["hold"],
        1, cv2.LINE_AA,
    )

    if record is not None:
        residual = record["residual"]
        readings = [
            f"confidence  {record['confidence']:.2f}",
            f"rigidity    {residual:.0%}" if np.isfinite(residual) else "rigidity    n/a",
            f"palm facing {record['palm_facing']:.2f}",
            f"off-frame   {record['out_of_frame']:.0%}",
        ]
        for line, text in enumerate(readings):
            cv2.putText(
                canvas, text, (x0 + 10, y0 + 46 + 17 * line), FONT, 0.40, COLOURS["dim"], 1, cv2.LINE_AA
            )

        # Per-finger confidence bars, coloured by what the state machine did.
        for row, (finger, state) in enumerate(record["states"].items()):
            top = y0 + 124 + 20 * row
            colour = COLOURS[state]
            score = record["finger_confidence"][row]
            cv2.putText(canvas, finger[:6], (x0 + 10, top + 11), FONT, 0.38, COLOURS["dim"], 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0 + 62, top), (x0 + 162, top + 12), COLOURS["grid"], -1)
            cv2.rectangle(
                canvas, (x0 + 62, top), (x0 + 62 + int(100 * min(max(score, 0.0), 1.0)), top + 12), colour, -1
            )
            cv2.putText(canvas, state, (x0 + 172, top + 11), FONT, 0.38, colour, 1, cv2.LINE_AA)

    box = min(height - 16, (width - 240) // 2 - 10)
    draw_hand(canvas, (x0 + 236, y0 + 8), box, raw_hand, COLOURS["raw"], "raw (MediaPipe)")
    draw_hand(
        canvas, (x0 + 246 + box, y0 + 8), box, stable_hand, COLOURS["stable"], "stabilised",
    )


def draw_chart(canvas, origin, width, height, label, records, frame_index, window, threshold) -> None:
    """Confidence and rigidity over time, with hold and blend runs shaded."""
    x0, y0 = origin
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), COLOURS["panel"], -1)
    # The state ribbon gets its own strip along the bottom rather than being
    # shaded behind the traces: on a hand that is held most of the clip, a
    # background band covers the whole chart and hides what you came to read.
    ribbon_height = 14
    plot_top = y0 + 18
    plot_bottom = y0 + height - 22 - ribbon_height
    plot_height = plot_bottom - plot_top
    ribbon_top = plot_bottom + 4

    start = max(0, frame_index - window)
    end = min(len(records), start + window * 2)
    if end <= start:
        return
    scale = width / float(end - start)

    def to_x(index: int) -> int:
        return int(x0 + (index - start) * scale)

    def to_y(value: float) -> int:
        return int(plot_bottom - min(max(value, 0.0), 1.0) * plot_height)

    # One column per frame, coloured by the worst state any finger is in and
    # scaled by how many fingers are in it.
    for index in range(start, end):
        states = list(records[index]["states"].values())
        affected = sum(1 for state in states if state in ("hold", "blend"))
        if not affected:
            continue
        colour = COLOURS["hold"] if "hold" in states else COLOURS["blend"]
        fraction = affected / float(len(states))
        band = max(int(ribbon_height * fraction), 2)
        cv2.rectangle(
            canvas,
            (to_x(index), ribbon_top + ribbon_height - band),
            (max(to_x(index + 1), to_x(index) + 1), ribbon_top + ribbon_height),
            colour,
            -1,
        )
    cv2.rectangle(
        canvas, (x0, ribbon_top), (x0 + width, ribbon_top + ribbon_height), COLOURS["grid"], 1
    )

    for value in (0.0, 0.5, 1.0):
        cv2.line(canvas, (x0, to_y(value)), (x0 + width, to_y(value)), COLOURS["grid"], 1)
    cv2.line(canvas, (x0, to_y(threshold)), (x0 + width, to_y(threshold)), COLOURS["threshold"], 1)

    confidence = [(to_x(i), to_y(records[i]["confidence"])) for i in range(start, end)]
    cv2.polylines(canvas, [np.array(confidence)], False, COLOURS["text"], 1, cv2.LINE_AA)

    # Rigidity residual on the same axes, scaled so 50% deviation is full height.
    # Broken into runs rather than drawn continuously: an untracked frame has no
    # residual at all, and joining across one would draw a flat line at the top
    # that reads as "terrible" when it means "not measured".
    run: list = []
    for index in range(start, end):
        value = records[index]["residual"]
        if np.isfinite(value):
            run.append((to_x(index), to_y(min(value, 0.5) / 0.5)))
            continue
        if len(run) > 1:
            cv2.polylines(canvas, [np.array(run)], False, COLOURS["residual"], 1, cv2.LINE_AA)
        run = []
    if len(run) > 1:
        cv2.polylines(canvas, [np.array(run)], False, COLOURS["residual"], 1, cv2.LINE_AA)

    cv2.line(
        canvas,
        (to_x(frame_index), plot_top),
        (to_x(frame_index), ribbon_top + ribbon_height),
        COLOURS["text"],
        1,
    )

    cv2.putText(canvas, f"{label} hand", (x0 + 8, y0 + 13), FONT, 0.42, COLOURS["text"], 1, cv2.LINE_AA)
    for offset, (text, colour) in enumerate(
        [
            ("confidence", COLOURS["text"]),
            ("rigidity (0-50%)", COLOURS["residual"]),
            (f"threshold {threshold:.2f}", COLOURS["threshold"]),
            ("held", COLOURS["hold"]),
            ("blending", COLOURS["blend"]),
        ]
    ):
        cv2.putText(
            canvas, text, (x0 + 110 + 118 * offset, y0 + 13), FONT, 0.38, colour, 1, cv2.LINE_AA
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    debug = json.loads(args.debug.expanduser().resolve().read_text(encoding="utf-8"))
    hands = debug.get("hands") or {}
    if not hands:
        raise SystemExit(f"{args.debug} records no hands.")

    labels = [label for label in ("Left", "Right") if label in hands]
    records = {label: hands[label]["frames"] for label in labels}
    frame_count = max(len(rows) for rows in records.values())
    fps = float(debug.get("fps") or 30.0)
    threshold = float(debug.get("threshold", 0.5))

    # Stage 2 trims the lead-in, so debug frame 0 is not video frame 0.
    offset = int((debug.get("trim") or {}).get("dropped_before", 0))

    raw_tracks = load_hands(args.before.expanduser().resolve() if args.before else None)
    stable_tracks = load_hands(args.after.expanduser().resolve() if args.after else None)

    capture = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open the video: {args.video}")

    panel_height = VIDEO_SIZE // max(len(labels), 1)
    width = VIDEO_SIZE + PANEL_WIDTH
    height = VIDEO_SIZE + CHART_HEIGHT * len(labels)

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # An OpenCV build without the mp4v encoder still hands back a writer; it just
    # drops every frame. Without this check the script would report success and
    # leave no video behind.
    if not writer.isOpened():
        capture.release()
        raise SystemExit(
            f"OpenCV could not open a video writer for {out_path} (mp4v codec unavailable in this build?)"
        )
    window = max(int(args.chart_seconds * fps / 2), 10)

    written = 0
    try:
        for index in range(frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index + offset)
            ok, frame = capture.read()
            canvas = np.full((height, width, 3), 22, dtype=np.uint8)
            if ok:
                canvas[:VIDEO_SIZE, :VIDEO_SIZE] = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE))
            cv2.putText(
                canvas,
                f"frame {index + offset}  ({(index + offset) / fps:.2f}s)",
                (10, VIDEO_SIZE - 12), FONT, 0.45, COLOURS["text"], 1, cv2.LINE_AA,
            )

            for row, label in enumerate(labels):
                rows = records[label]
                record = rows[index] if index < len(rows) else None
                draw_hand_panel(
                    canvas,
                    (VIDEO_SIZE, row * panel_height),
                    PANEL_WIDTH,
                    panel_height,
                    label,
                    record,
                    (raw_tracks.get(label) or {}).get(index + offset),
                    (stable_tracks.get(label) or {}).get(index),
                )
                draw_chart(
                    canvas,
                    (0, VIDEO_SIZE + row * CHART_HEIGHT),
                    width,
                    CHART_HEIGHT,
                    label,
                    rows,
                    index,
                    window,
                    threshold,
                )

            writer.write(canvas)
            written += 1
    finally:
        capture.release()
        writer.release()

    print(f"[overlay] wrote {out_path} ({written} frames, {width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
