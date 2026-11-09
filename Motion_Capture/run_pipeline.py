"""One video in, one animated Blender scene out.

    python run_pipeline.py --video ../videos/Water.mp4

That runs all three stages:

    stage 1    extract_landmarks.py     (needs a Python with mediapipe: .venv311)
    stage 2    postprocess_landmarks.py (needs a Python with skellyforge: .venv)
    stage 2.5  stabilize_hands.py       (needs numpy and PyYAML only)
    stage 3    blender_retarget.py      (needs Blender, runs headless)

Everything lands in ``output/<video name>/``: ``landmarks.json``, the ``.blend``,
and whatever extra exports were asked for. That folder is git-ignored - it holds
landmark data derived from footage of a real person, which must never be
committed (see CLAUDE.md).

This script only orchestrates; it deliberately imports neither mediapipe nor bpy,
because no single interpreter on this machine has both.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_MODEL = REPO_ROOT / "Y Bot.fbx"
DEFAULT_OUTPUT_ROOT = HERE / "output"


def find_mediapipe_python(explicit: Path | None) -> Path:
    """Locate an interpreter that can import mediapipe."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("MEDIAPIPE_PYTHON"):
        candidates.append(Path(os.environ["MEDIAPIPE_PYTHON"]))
    candidates += [
        REPO_ROOT / "venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import mediapipe"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "No Python with mediapipe installed was found. Point --python at one, or set "
        "MEDIAPIPE_PYTHON. In this repo, .venv311 has it (mediapipe 0.10.9 has no "
        "wheel for the 3.14 interpreter in .venv)."
    )


def find_skellyforge_python(explicit: Path | None) -> Path:
    """Locate an interpreter that can import skellyforge (freemocap's post-processing)."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("SKELLYFORGE_PYTHON"):
        candidates.append(Path(os.environ["SKELLYFORGE_PYTHON"]))
    candidates += [
        Path(sys.executable),
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / "venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "bin" / "python",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run([str(candidate), "-c", "import skellyforge"], capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "No Python with skellyforge installed was found, so stage 2 cannot run. Install it:\n"
        '    .venv/Scripts/python.exe -m pip install "skellyforge @ git+https://github.com/freemocap/skellyforge@main"\n'
        "or pass --postprocess-python, set SKELLYFORGE_PYTHON, or skip the stage with --no-postprocess."
    )


def find_stabilize_python(explicit: Path | None) -> Path:
    """Locate an interpreter for stage 2.5. It needs only numpy and PyYAML."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates += [
        Path(sys.executable),
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / "venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv311" / "bin" / "python",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import numpy, yaml"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "No Python with numpy and PyYAML was found, so stage 2.5 cannot run. Install them "
        "into one of this repo's virtualenvs, or skip the stage with --no-stabilize."
    )


def blender_version_key(executable: Path) -> tuple:
    """Numeric version of a Blender install, read from its folder name.

    The installer names the folder "Blender 4.2"; comparing those names as
    strings would put 4.10 before 4.2, so the digits are compared as numbers.
    A folder with no version in its name sorts before every versioned one.
    """
    return tuple(int(part) for part in re.findall(r"\d+", executable.parent.name))


def find_blender(explicit: Path | None) -> Path:
    """Locate the Blender executable, preferring the newest install found."""
    if explicit is not None:
        if not Path(explicit).is_file():
            raise SystemExit(f"Blender not found at {explicit}")
        return Path(explicit)
    if os.environ.get("BLENDER_EXE"):
        candidate = Path(os.environ["BLENDER_EXE"])
        if candidate.is_file():
            return candidate

    if sys.platform.startswith("win"):
        found = []
        for drive in ("C:/", "D:/"):
            base = Path(drive) / "Program Files" / "Blender Foundation"
            if base.is_dir():
                found += base.glob("*/blender.exe")
        if found:
            # Sort across every drive at once, so an older install on D: cannot
            # win just because it was listed last.
            return max(found, key=blender_version_key)
    elif sys.platform == "darwin":
        candidate = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if candidate.is_file():
            return candidate
    else:
        for candidate in (Path("/usr/bin/blender"), Path("/usr/local/bin/blender"), Path("/snap/bin/blender")):
            if candidate.is_file():
                return candidate
    raise SystemExit("Could not find Blender. Pass --blender /path/to/blender, or set BLENDER_EXE.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn one video into a baked Blender animation on the avatar model.",
    )
    parser.add_argument("--video", required=True, type=Path, help="The single input video.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Rigged FBX to animate.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: output/<video name>/).")
    parser.add_argument("--python", type=Path, default=None, help="Python interpreter that has mediapipe.")
    parser.add_argument("--blender", type=Path, default=None, help="Blender executable.")
    parser.add_argument("--skip-extract", action="store_true", help="Reuse an existing landmarks.json.")
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=None,
        help="Use this landmarks.json instead of the one in the output folder (implies --skip-extract).",
    )
    parser.add_argument("--selfie", action="store_true", help="The footage is already mirrored (webcam selfie view).")
    parser.add_argument("--mirror", action="store_true", help="Mirror the retarget left/right.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help=(
            "Stage 1 exponential smoothing. Off by default because stage 2's zero-lag "
            "Butterworth filter does the job better; raise it only with --no-postprocess."
        ),
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip freemocap's smoothing and gap-filling (stage 2) and retarget the raw landmarks.",
    )
    parser.add_argument(
        "--filter-cutoff",
        type=float,
        default=6.0,
        help="Butterworth cutoff in Hz for stage 2 (freemocap's default is 6.0). Lower is smoother.",
    )
    parser.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order for stage 2.")
    parser.add_argument(
        "--postprocess-python",
        type=Path,
        default=None,
        help="Python interpreter that has skellyforge, if it is not the one running this script.",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help=(
            "Skip stage 2.5. The fingers then keep whatever MediaPipe reported, including the "
            "frames where it was guessing - no occlusion handling, no joint limits."
        ),
    )
    parser.add_argument(
        "--stabilize-config",
        type=Path,
        default=None,
        help="stabilize.yaml for stage 2.5 (default: the one beside the script). All tuning is in it.",
    )
    parser.add_argument(
        "--debug-hands",
        action="store_true",
        help=(
            "Write the stage 2.5 trace and render debug_hands.mp4: confidence, hold state and "
            "rigidity residual, frame-synced to the source video."
        ),
    )
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Flatten noisy monocular depth for the body.")
    parser.add_argument("--max-frames", type=int, default=0, help="Only process the first N frames.")
    parser.add_argument("--no-hands", action="store_true", help="Body and wrists only, no finger solve.")
    parser.add_argument("--no-head", action="store_true", help="Leave the head and neck in the rest pose.")
    parser.add_argument(
        "--no-roll-calibration",
        action="store_true",
        help="Keep MediaPipe's head roll as measured, bias and all.",
    )
    parser.add_argument("--export-fbx", action="store_true", help="Also export an animated FBX.")
    parser.add_argument("--export-glb", action="store_true", help="Also export a .glb.")
    parser.add_argument("--debug-video", action="store_true", help="Also write the annotated MediaPipe overlay video.")
    parser.add_argument("--preview", action="store_true", help="Also render a few PNG frames to eyeball the result.")
    parser.add_argument("--keep-helpers", action="store_true", help="Leave the target empties in the .blend.")
    return parser.parse_args(argv)


def run(step: str, command: list[str]) -> None:
    print(f"\n=== {step} ===\n{' '.join(command)}\n", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    model = args.model.expanduser().resolve()
    output_dir = (args.out.expanduser().resolve() if args.out else DEFAULT_OUTPUT_ROOT / video.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    landmarks_path = args.landmarks.expanduser().resolve() if args.landmarks else output_dir / "landmarks.json"
    # Later stages rebind landmarks_path to their own output; the debug overlay
    # still needs the unprocessed stage 1 file, wherever it actually came from.
    raw_landmarks_path = landmarks_path
    blend_path = output_dir / f"{video.stem}.blend"

    if args.skip_extract or args.landmarks:
        if not landmarks_path.is_file():
            raise SystemExit(f"Extraction was skipped but {landmarks_path} does not exist.")
        print(f"[pipeline] reusing {landmarks_path}")
    else:
        python_exe = find_mediapipe_python(args.python)
        print(f"[pipeline] mediapipe python: {python_exe}")
        extract_command = [
            str(python_exe),
            str(HERE / "extract_landmarks.py"),
            "--video", str(video),
            "--out", str(landmarks_path),
            "--smoothing", str(args.smoothing),
        ]
        if args.selfie:
            extract_command.append("--selfie")
        if args.max_frames:
            extract_command += ["--max-frames", str(args.max_frames)]
        if args.debug_video:
            extract_command += ["--debug-video", str(output_dir / "debug_overlay.mp4")]
        run("stage 1: extract landmarks", extract_command)

    if not args.no_postprocess:
        postprocess_python = find_skellyforge_python(args.postprocess_python)
        print(f"[pipeline] skellyforge python: {postprocess_python}")
        filtered_path = output_dir / "landmarks_filtered.json"
        postprocess_command = [
            str(postprocess_python),
            str(HERE / "postprocess_landmarks.py"),
            "--landmarks", str(landmarks_path),
            "--out", str(filtered_path),
            "--cutoff", str(args.filter_cutoff),
            "--order", str(args.filter_order),
        ]
        if not args.no_stabilize:
            # Blank unreliable hands here, before filtfilt can smear them across
            # their neighbours, and leave the finger shape for stage 2.5's One
            # Euro filter rather than low-passing it twice.
            postprocess_command += ["--confidence-gate", "--hand-shape-filter", "none"]
            if args.stabilize_config:
                postprocess_command += ["--stabilize-config", str(args.stabilize_config)]
        run("stage 2: smooth and gap-fill (freemocap/skellyforge)", postprocess_command)
        landmarks_path = filtered_path

    if not args.no_stabilize:
        stabilize_python = find_stabilize_python(None)
        print(f"[pipeline] stabiliser python: {stabilize_python}")
        stabilized_path = output_dir / "landmarks_stabilized.json"
        debug_path = output_dir / "stabilize_debug.json"
        stabilize_command = [
            str(stabilize_python),
            str(HERE / "stabilize_hands.py"),
            "--landmarks", str(landmarks_path),
            "--out", str(stabilized_path),
        ]
        if args.stabilize_config:
            stabilize_command += ["--config", str(args.stabilize_config)]
        if args.debug_hands:
            stabilize_command += ["--debug-out", str(debug_path)]
        run("stage 2.5: occlusion handling and joint limits", stabilize_command)
        landmarks_path = stabilized_path

        if args.debug_hands:
            run(
                "stage 2.5b: debug overlay",
                [
                    str(stabilize_python),
                    str(HERE / "debug_overlay.py"),
                    "--video", str(video),
                    "--debug", str(debug_path),
                    "--out", str(output_dir / "debug_hands.mp4"),
                    # The raw and stabilised wireframes side by side are the
                    # point of the overlay - a finger snapping into an
                    # impossible bend is obvious there and invisible in a chart.
                    "--before", str(raw_landmarks_path),
                    "--after", str(stabilized_path),
                ],
            )

    blender_exe = find_blender(args.blender)
    print(f"[pipeline] blender: {blender_exe}")
    retarget_command = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        # Without this, Blender exits 0 even when the script raised, and a failed
        # retarget would look like a successful run.
        "--python-exit-code", "1",
        "--python", str(HERE / "blender_retarget.py"),
        "--",
        "--landmarks", str(landmarks_path),
        "--model", str(model),
        "--blend-out", str(blend_path),
        "--depth-scale", str(args.depth_scale),
    ]
    if args.mirror:
        retarget_command.append("--mirror")
    for flag in ("--no-hands", "--no-head", "--no-roll-calibration"):
        if getattr(args, flag.lstrip("-").replace("-", "_")):
            retarget_command.append(flag)
    if args.keep_helpers:
        retarget_command.append("--keep-helpers")
    if args.export_fbx:
        retarget_command += ["--export-fbx", str(output_dir / f"{video.stem}_animated.fbx")]
    if args.export_glb:
        retarget_command += ["--export-glb", str(output_dir / f"{video.stem}_animated.glb")]
    if args.preview:
        retarget_command += ["--render-preview", str(output_dir / "preview")]
    run("stage 3: retarget in blender", retarget_command)

    print(f"\n[pipeline] done - open {blend_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
