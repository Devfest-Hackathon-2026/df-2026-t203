"""FastAPI Web Server for Ninjoucon Motion Capture & Copy-Training Studio.

Integrates:
1. Teacher Video management (MP4 upload or YouTube link download via yt-dlp).
2. Keypose extraction (`utilities/preprocess.py` -> `generate_keyposes_new`).
3. Webcam overlay and interactive copy-training via WebSocket `/ws/training/{video_id}`.
4. Clean user webcam recording (without overlays) tucked in a per-video
   subdirectory in the sidebar, with download & Motion Capture 3D processing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Ensure workspace root is on sys.path so `utilities` and `training_session` import cleanly
WEB_APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = WEB_APP_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

from training_session import build_keypose_sync_for_recording, run_web_training_session  # noqa: E402
from browser_training import run_browser_training  # noqa: E402
from utilities.keyposes import recording_keypose_times  # noqa: E402

DATA_DIR = Path(os.environ["MOTIONLEARN_DATA_DIR"]).resolve() if "MOTIONLEARN_DATA_DIR" in os.environ else None
DEFAULT_DIST = WEB_APP_DIR / "dist"
PRODUCT_DIST = Path(os.environ["MOTIONLEARN_DIST"]).resolve() if "MOTIONLEARN_DIST" in os.environ else (DEFAULT_DIST if (DEFAULT_DIST / "index.html").is_file() else None)
PUBLIC_ORIGIN = os.getenv("MOTIONLEARN_ORIGIN")
RECORDINGS_DATA_DIR = (DATA_DIR or WEB_APP_DIR) / "recordings_data"
TEACHER_VIDEOS_DIR = (DATA_DIR or WEB_APP_DIR) / "teacher_videos"
MOTION_CAPTURE_DIR = WORKSPACE_DIR / "Motion_Capture"
PIPELINE_SCRIPT = MOTION_CAPTURE_DIR / "run_pipeline.py"
OUTPUTS_DIR = (DATA_DIR or MOTION_CAPTURE_DIR) / "output"
PYTHON_EXE = WORKSPACE_DIR / "venv" / "Scripts" / "python.exe"
DOJO_GLB_PATH = WORKSPACE_DIR / "Dojo.glb"

RECORDINGS_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RECORDINGS_DATA_DIR / "recordings.json"
VIDEOS_META_FILE = TEACHER_VIDEOS_DIR / "videos.json"

@asynccontextmanager
async def lifespan(_app):
    if DATA_DIR is not None:
        if not VIDEOS_META_FILE.exists():
            seed = WEB_APP_DIR / "teacher_videos"
            shutil.copy2(seed / "vid_sample.mp4", TEACHER_VIDEOS_DIR)
            shutil.copytree(seed / "vid_sample_keyposes", TEACHER_VIDEOS_DIR / "vid_sample_keyposes", dirs_exist_ok=True)
            shutil.copy2(seed / "videos.json", VIDEOS_META_FILE)
        if not METADATA_FILE.exists():
            save_metadata_locked([])
    with metadata_lock:
        for load, save in ((load_videos_locked, save_videos_locked), (load_recordings_locked, save_metadata_locked)):
            items = load()
            interrupted = [item for item in items if item.get("status") == "processing"]
            for item in interrupted:
                item.update(status="error", error_message="Processing was interrupted. Retry to continue.")
            if interrupted:
                save(items)
    for directory in RECORDINGS_DATA_DIR.glob(".take-*"):
        if directory.is_dir():
            shutil.rmtree(directory)
    yield


app = FastAPI(title="MotionLearn", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if PUBLIC_ORIGIN else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_and_api_cache(request, call_next):
    origin = request.headers.get("origin")
    if PUBLIC_ORIGIN and origin and origin != PUBLIC_ORIGIN and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        return JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/training", "/analysis"} or path.startswith(("/static/", "/api/")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# In-memory logs and active process tracking
active_logs: Dict[str, List[str]] = {}
active_stages: Dict[str, str] = {}
video_keypose_logs: Dict[str, List[str]] = {}
jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motionlearn")
metadata_lock = threading.RLock()
active_browser_videos: set[str] = set()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


class YouTubeRequest(BaseModel):
    url: str
    name: str | None = None


def convert_to_browser_mp4(src_path: Path, dest_path: Path, *, strict=False) -> None:
    """Convert/remux any video file to browser-compatible H.264 MP4."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        try:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(src_path),
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                str(dest_path),
            ]
            subprocess.run(cmd, check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if src_path != dest_path:
                src_path.unlink(missing_ok=True)
            return
        except Exception:
            pass
    if strict:
        dest_path.unlink(missing_ok=True)
        src_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not convert the video to MP4. Check the file and try again.")
    if src_path != dest_path:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        src_path.rename(dest_path)


# ---------------------------------------------------------------------------
# Metadata Helpers (Teacher Videos & Per-Video Recordings)
# ---------------------------------------------------------------------------
def load_videos_locked() -> List[Dict[str, Any]]:
    if not VIDEOS_META_FILE.exists():
        sample_video = WORKSPACE_DIR / "captured_vid.mp4"
        initial_videos: List[Dict[str, Any]] = []
        if sample_video.exists():
            dest = TEACHER_VIDEOS_DIR / "vid_sample.mp4"
            if not dest.exists():
                shutil.copy2(sample_video, dest)
            initial_videos.append({
                "id": "vid_sample",
                "name": "Sample Video (captured_vid)",
                "source_type": "mp4",
                "source_url": None,
                "filename": "vid_sample.mp4",
                "created_at": datetime.now().strftime("%b %d, %Y, %I:%M %p"),
                "is_processed": False,
                "status": "unprocessed",
                "keyposes_dir": "vid_sample_keyposes",
                "keypose_count": 0,
                "timestamps": {},
            })
        VIDEOS_META_FILE.write_text(json.dumps(initial_videos, indent=2), encoding="utf-8")
        return initial_videos

    try:
        return json.loads(VIDEOS_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_videos_locked(videos: List[Dict[str, Any]]) -> None:
    write_json(VIDEOS_META_FILE, videos)


def renumber_recordings_locked(items: List[Dict[str, Any]]) -> None:
    """Renumber recordings sequentially starting from 1 within each teacher video (or misc_mocap)."""
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for entry in items:
        vid_id = entry.get("video_id") or "misc_mocap"
        entry["video_id"] = vid_id
        by_video.setdefault(vid_id, []).append(entry)

    for vid_id, group in by_video.items():
        total = len(group)
        for position, entry in enumerate(group):
            number = total - position
            entry["recording_no"] = number
            entry["title"] = f"Recording #{number} - {entry.get('created_at', '')}"


def load_recordings_locked() -> List[Dict[str, Any]]:
    if not METADATA_FILE.exists():
        sample_video = WORKSPACE_DIR / "captured_vid.mp4"
        initial: List[Dict[str, Any]] = []
        if sample_video.exists():
            sample_dest = RECORDINGS_DATA_DIR / "rec_sample.mp4"
            if not sample_dest.exists():
                shutil.copy2(sample_video, sample_dest)

            test_glb = OUTPUTS_DIR / "test" / "captured_vid_animated.glb"
            is_proc = test_glb.exists()
            now_str = datetime.now().strftime("%b %d, %Y, %I:%M %p")

            initial.append({
                "id": "rec_sample",
                "video_id": "misc_mocap",
                "video_name": "Test Motion Capture (Misc.)",
                "recording_no": 1,
                "title": f"Recording #1 - {now_str}",
                "created_at": now_str,
                "filename": "rec_sample.mp4",
                "is_processed": is_proc,
                "status": "completed" if is_proc else "unprocessed",
                "output_dir_name": "test" if is_proc else "rec_sample",
                "glb_filename": "captured_vid_animated.glb" if is_proc else None,
                "blend_filename": "captured_vid.blend" if is_proc else None,
                "fbx_filename": "captured_vid_animated.fbx" if is_proc else None,
            })
        METADATA_FILE.write_text(json.dumps(initial, indent=2), encoding="utf-8")
        return initial

    try:
        items = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        videos_by_id = {v["id"]: v for v in load_videos_locked()}
        changed = False
        for item in items:
            vid_id = item.get("video_id")
            # Move legacy or unprocessed-video recordings to misc_mocap so unprocessed videos have no recordings
            if not vid_id or (vid_id != "misc_mocap" and not videos_by_id.get(vid_id, {}).get("is_processed", False)):
                item["video_id"] = "misc_mocap"
                item["video_name"] = "Test Motion Capture (Misc.)"
                item["teacher_video_filename"] = None
                changed = True
            elif vid_id != "misc_mocap":
                parent_vid = videos_by_id.get(vid_id)
                if parent_vid:
                    if item.get("teacher_video_filename") != parent_vid.get("filename"):
                        item["teacher_video_filename"] = parent_vid.get("filename")
                        changed = True
                    if not item.get("browser_take") and not item.get("keypose_sync") and parent_vid.get("keyposes_dir"):
                        rec_file = RECORDINGS_DATA_DIR / item.get("filename", "")
                        kp_dir = TEACHER_VIDEOS_DIR / parent_vid["keyposes_dir"]
                        item["keypose_sync"] = build_keypose_sync_for_recording(rec_file, kp_dir)
                        changed = True
        if changed:
            renumber_recordings_locked(items)
            METADATA_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return items
    except Exception:
        return []


def save_metadata_locked(data: List[Dict[str, Any]]) -> None:
    write_json(METADATA_FILE, data)


def load_metadata() -> List[Dict[str, Any]]:
    with metadata_lock:
        load_videos_locked()
        return load_recordings_locked()


def get_video_by_id(video_id: str) -> Dict[str, Any] | None:
    with metadata_lock:
        for v in load_videos_locked():
            if v["id"] == video_id:
                return v
    return None


def update_video(video_id: str, updates: Dict[str, Any]) -> Dict[str, Any] | None:
    with metadata_lock:
        videos = load_videos_locked()
        updated = None
        for v in videos:
            if v["id"] == video_id:
                v.update(updates)
                updated = v
                break
        if updated:
            save_videos_locked(videos)
        return updated


def get_recording_by_id(rec_id: str) -> Dict[str, Any] | None:
    with metadata_lock:
        for item in load_recordings_locked():
            if item["id"] == rec_id:
                return item
    return None


def update_recording(rec_id: str, updates: Dict[str, Any]) -> Dict[str, Any] | None:
    with metadata_lock:
        items = load_recordings_locked()
        updated_item = None
        for item in items:
            if item["id"] == rec_id:
                item.update(updates)
                updated_item = item
                break
        if updated_item:
            save_metadata_locked(items)
        return updated_item


def register_new_recording(
    video_id: str,
    video_name: str,
    final_path: Path,
    rec_id: str,
    formatted_date: str,
    keypose_sync: List[Dict[str, Any]] | None = None,
    feedback: list | None = None,
) -> Dict[str, Any]:
    with metadata_lock:
        items = load_recordings_locked()
        new_item = {
            "id": rec_id,
            "video_id": video_id,
            "video_name": video_name,
            "recording_no": 1,
            "title": f"Recording #1 - {formatted_date}",
            "created_at": formatted_date,
            "filename": final_path.name,
            "is_processed": False,
            "status": "unprocessed",
            "output_dir_name": rec_id,
            "glb_filename": None,
            "blend_filename": None,
            "fbx_filename": None,
            "keypose_sync": keypose_sync or [],
        }
        items.insert(0, new_item)
        renumber_recordings_locked(items)
        feedback_file = RECORDINGS_DATA_DIR / f"{rec_id}.feedback.json"
        if feedback is not None:
            new_item["browser_take"] = True
            write_json(feedback_file, feedback)
        try:
            save_metadata_locked(items)
        except OSError:
            feedback_file.unlink(missing_ok=True)
            raise
        return new_item


# ---------------------------------------------------------------------------
# Background Workers (Keypose Extraction & Mocap Pipeline)
# ---------------------------------------------------------------------------
def run_keyposes_worker(video_id: str, video_path: Path, keyposes_dir: Path) -> None:
    video_keypose_logs[video_id] = [f"[init] Extracting keyposes from {video_path.name}..."]

    out_keyposes = keyposes_dir / "teacher_keyposes.json"
    out_timestamps = keyposes_dir / "teacher_timestamps.json"

    try:
        from utilities.preprocess import generate_keyposes_new

        keyposes_dir.mkdir(parents=True, exist_ok=True)
        for path in (out_keyposes, out_timestamps, keyposes_dir / "teacher_trajectories.json"):
            path.unlink(missing_ok=True)
        video_keypose_logs[video_id].append("[mediapipe] Analyzing full-body velocity and movement peaks...")
        generate_keyposes_new(
            str(video_path),
            output_path=str(out_keyposes),
            output_path_time=str(out_timestamps),
            frame_skip=2,
        )

        if out_timestamps.exists() and out_keyposes.exists():
            raw_ts = json.loads(out_timestamps.read_text(encoding="utf-8"))
            if not raw_ts:
                raise ValueError("No usable keyposes were detected.")
            timestamps = {str(k): float(v) for k, v in raw_ts.items()}

            video_keypose_logs[video_id].append(
                f"[success] Extracted {len(timestamps)} keyposes! Ready for copy-training."
            )
            update_video(video_id, {
                "is_processed": True,
                "status": "ready",
                "keypose_count": len(timestamps),
                "timestamps": timestamps,
            })
        else:
            video_keypose_logs[video_id].append("[error] Not enough valid pose frames detected in video.")
            update_video(video_id, {"status": "error", "error_message": "No usable keyposes were detected."})
    except Exception as exc:
        video_keypose_logs[video_id].append(f"[exception] {exc}")
        update_video(video_id, {"status": "error", "error_message": str(exc)})


def run_pipeline_worker(
    rec_id: str, video_path: Path, output_dir: Path, previous_output_dir: Path | None = None,
) -> None:
    active_logs[rec_id] = [f"[init] Starting motion capture pipeline for {video_path.name}..."]
    active_stages[rec_id] = "Starting..."

    cmd = [
        str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable),
        str(PIPELINE_SCRIPT),
        "--video",
        str(video_path),
        "--out",
        str(output_dir),
        "--mirror",
        "--export-glb",
        "--export-fbx",
    ]

    active_logs[rec_id].append(f"[exec] {' '.join(cmd)}")

    published = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(MOTION_CAPTURE_DIR),
        )

        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            if line_str:
                active_logs[rec_id].append(line_str)
                if len(active_logs[rec_id]) > 500:
                    active_logs[rec_id] = active_logs[rec_id][-400:]

                if "stage 1" in line_str.lower():
                    active_stages[rec_id] = "Stage 1/3: Extracting Pose & Hand Landmarks"
                elif "stage 2.5" in line_str.lower():
                    active_stages[rec_id] = "Stage 2.5: Stabilizing Hands & Joint Limits"
                elif "stage 2" in line_str.lower():
                    active_stages[rec_id] = "Stage 2/3: Smoothing & Gap-Filling Body"
                elif "stage 3" in line_str.lower():
                    active_stages[rec_id] = "Stage 3/3: Retargeting in Blender onto Y Bot"

        process.wait()

        glb_files = list(output_dir.glob("*.glb"))
        blend_files = list(output_dir.glob("*.blend"))
        fbx_files = list(output_dir.glob("*.fbx"))
        has_glb = len(glb_files) > 0

        if process.returncode == 0 and has_glb:
            glb_name = glb_files[0].name
            blend_name = blend_files[0].name if blend_files else None
            fbx_name = fbx_files[0].name if fbx_files else None

            rec_item = get_recording_by_id(rec_id) or {}
            keypose_sync = rec_item.get("keypose_sync") or []
            vid_id = rec_item.get("video_id")
            if vid_id and vid_id != "misc_mocap" and not keypose_sync and not rec_item.get("browser_take"):
                parent_vid = get_video_by_id(vid_id)
                if parent_vid and parent_vid.get("keyposes_dir"):
                    kp_dir = TEACHER_VIDEOS_DIR / parent_vid["keyposes_dir"]
                    keypose_sync = build_keypose_sync_for_recording(video_path, kp_dir)

            active_stages[rec_id] = "Completed"
            active_logs[rec_id].append(f"[success] Pipeline finished! Generated {glb_name}")

            published = update_recording(rec_id, {
                "is_processed": True,
                "status": "completed",
                "output_dir_name": output_dir.name,
                "mocap_mirrored": True,
                "glb_filename": glb_name,
                "blend_filename": blend_name,
                "fbx_filename": fbx_name,
                "keypose_sync": keypose_sync,
            }) is not None
            if (
                published
                and previous_output_dir is not None
                and previous_output_dir.name == rec_id
                and previous_output_dir.resolve().parent == OUTPUTS_DIR.resolve()
                and not previous_output_dir.is_symlink()
            ):
                shutil.rmtree(previous_output_dir, ignore_errors=True)
        else:
            err = f"Pipeline exited with code {process.returncode} (glb created: {has_glb})"
            active_stages[rec_id] = "Failed"
            active_logs[rec_id].append(f"[error] {err}")
            update_recording(rec_id, {
                "status": "error",
                "error_message": err,
            })

    except Exception as exc:
        active_stages[rec_id] = "Error"
        active_logs[rec_id].append(f"[exception] {str(exc)}")
        update_recording(rec_id, {
            "status": "error",
            "error_message": str(exc),
        })
    finally:
        if (
            previous_output_dir is not None
            and not published
            and output_dir.resolve().parent == OUTPUTS_DIR.resolve()
            and not output_dir.is_symlink()
        ):
            shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Teacher Video API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/videos")
def list_videos():
    with metadata_lock:
        videos = load_videos_locked()
        recordings = load_recordings_locked()
        result = []
        for v in videos:
            v_copy = dict(v)
            v_copy["recordings"] = [r for r in recordings if r.get("video_id") == v["id"]]
            result.append(v_copy)
        return result


@app.post("/api/videos/upload")
async def upload_teacher_video(
    video: UploadFile = File(...),
    name: str | None = Form(None),
    auto_process: bool = False,
):
    timestamp = datetime.now()
    vid_id = f"vid_{uuid4().hex}"
    formatted_date = timestamp.strftime("%b %d, %Y, %I:%M %p")

    raw_name = Path(video.filename or "video.mp4").stem
    display_name = (name or "").strip() or raw_name

    raw_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    temp_path = TEACHER_VIDEOS_DIR / f"{vid_id}_raw{raw_ext}"
    final_path = TEACHER_VIDEOS_DIR / f"{vid_id}.mp4"

    await run_in_threadpool(store_upload, video, temp_path, final_path, auto_process)

    new_video = {
        "id": vid_id,
        "name": display_name,
        "source_type": "mp4",
        "source_url": None,
        "filename": final_path.name,
        "created_at": formatted_date,
        "is_processed": False,
        "status": "unprocessed",
        "keyposes_dir": f"{vid_id}_keyposes",
        "keypose_count": 0,
        "timestamps": {},
    }

    with metadata_lock:
        videos = load_videos_locked()
        videos.insert(0, new_video)
        save_videos_locked(videos)

    new_video["recordings"] = []
    if auto_process:
        process_teacher_video_keyposes(vid_id)
        new_video["status"] = "processing"
    return new_video


@app.post("/api/videos/youtube")
def add_youtube_video(req: YouTubeRequest, auto_process: bool = False):
    url = req.url.strip()
    try:
        parsed = urlparse(url)
        valid = parsed.scheme in {"https", "http"} and parsed.netloc.lower() in {
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
        }
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(status_code=400, detail="Enter a YouTube video URL.")

    timestamp = datetime.now()
    vid_id = f"vid_{uuid4().hex}"
    formatted_date = timestamp.strftime("%b %d, %Y, %I:%M %p")
    final_path = TEACHER_VIDEOS_DIR / f"{vid_id}.mp4"

    try:
        import yt_dlp

        outtmpl = str(TEACHER_VIDEOS_DIR / f"{vid_id}_dl.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "allowed_extractors": ["youtube"],
            "max_filesize": 100 * 1024 * 1024,
            "socket_timeout": 20,
            "retries": 1,
        }
        if shutil.which("node"):
            ydl_opts["js_runtimes"] = {"node": {}}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            title = (req.name or "").strip() or info_dict.get("title") or f"YouTube Video ({vid_id})"
            if "requested_downloads" in info_dict and info_dict["requested_downloads"]:
                downloaded_file = Path(info_dict["requested_downloads"][0]["filepath"])
            else:
                downloaded_file = Path(ydl.prepare_filename(info_dict))
    except Exception as exc:
        for partial in TEACHER_VIDEOS_DIR.glob(f"{vid_id}_dl.*"):
            partial.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to download YouTube video: {exc}") from exc

    if not downloaded_file.exists():
        matches = list(TEACHER_VIDEOS_DIR.glob(f"{vid_id}_dl.*"))
        if not matches:
            raise HTTPException(status_code=500, detail="Downloaded YouTube file not found on disk")
        downloaded_file = matches[0]

    try:
        convert_to_browser_mp4(downloaded_file, final_path, strict=auto_process)
    finally:
        for partial in TEACHER_VIDEOS_DIR.glob(f"{vid_id}_dl.*"):
            partial.unlink(missing_ok=True)

    new_video = {
        "id": vid_id,
        "name": title,
        "source_type": "youtube",
        "source_url": url,
        "filename": final_path.name,
        "created_at": formatted_date,
        "is_processed": False,
        "status": "unprocessed",
        "keyposes_dir": f"{vid_id}_keyposes",
        "keypose_count": 0,
        "timestamps": {},
    }

    with metadata_lock:
        videos = load_videos_locked()
        videos.insert(0, new_video)
        save_videos_locked(videos)

    new_video["recordings"] = []
    if auto_process:
        process_teacher_video_keyposes(vid_id)
        new_video["status"] = "processing"
    return new_video


@app.post("/api/videos/{video_id}/process")
def process_teacher_video_keyposes(video_id: str):
    with metadata_lock:
        video = get_video_by_id(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Teacher video not found")
        if video.get("status") in {"processing", "ready"}:
            return {"status": video["status"]}
        video_path = TEACHER_VIDEOS_DIR / video["filename"]
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Teacher video file missing")
        keyposes_dir = TEACHER_VIDEOS_DIR / video["keyposes_dir"]
        video_keypose_logs[video_id] = []
        update_video(video_id, {"status": "processing", "error_message": None})
        jobs.submit(run_keyposes_worker, video_id, video_path, keyposes_dir)
    return {"message": "Keypose extraction started", "status": "processing"}


@app.get("/api/videos/{video_id}/status")
def get_teacher_video_status(video_id: str):
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Teacher video not found")

    return {
        "id": video_id,
        "status": video.get("status", "unprocessed"),
        "is_processed": video.get("is_processed", False),
        "keypose_count": video.get("keypose_count", 0),
        "timestamps": video.get("timestamps", {}),
        "logs": video_keypose_logs.get(video_id, []),
        "error_message": video.get("error_message"),
    }


@app.delete("/api/videos/{video_id}")
def delete_teacher_video(video_id: str):
    with metadata_lock:
        video = get_video_by_id(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Teacher video not found")
        recordings = load_recordings_locked()
        if video_id in active_browser_videos or video.get("status") == "processing" or any(
            rec.get("video_id") == video_id and rec.get("status") == "processing" for rec in recordings
        ):
            raise HTTPException(status_code=409, detail="Cannot delete a teacher video while it is in use.")
        return delete_teacher_media(video, recordings)


def delete_teacher_media(video, recordings):
    video_id = video["id"]

    # Delete teacher video file and keyposes dir
    (TEACHER_VIDEOS_DIR / video["filename"]).unlink(missing_ok=True)
    kp_dir = TEACHER_VIDEOS_DIR / video["keyposes_dir"]
    if kp_dir.exists():
        shutil.rmtree(kp_dir, ignore_errors=True)

    with metadata_lock:
        videos = [v for v in load_videos_locked() if v["id"] != video_id]
        save_videos_locked(videos)

        kept_recordings = []
        for rec in recordings:
            if rec.get("video_id") == video_id:
                (RECORDINGS_DATA_DIR / rec["filename"]).unlink(missing_ok=True)
                (RECORDINGS_DATA_DIR / f"{rec['id']}.feedback.json").unlink(missing_ok=True)
                out_dir = OUTPUTS_DIR / rec["output_dir_name"]
                if out_dir.exists() and rec["output_dir_name"] != "test":
                    shutil.rmtree(out_dir, ignore_errors=True)
            else:
                kept_recordings.append(rec)
        renumber_recordings_locked(kept_recordings)
        save_metadata_locked(kept_recordings)

    return {"deleted": True, "id": video_id}


# ---------------------------------------------------------------------------
# Interactive Copy-Training WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/training/{video_id}")
async def training_websocket(websocket: WebSocket, video_id: str):
    browser = websocket.query_params.get("camera") == "browser"
    if PUBLIC_ORIGIN and websocket.headers.get("origin") != PUBLIC_ORIGIN:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    video = get_video_by_id(video_id)
    if not video:
        await websocket.send_json({"type": "ERROR", "message": "Teacher video not found."})
        await websocket.close()
        return

    if not video.get("is_processed", False):
        await websocket.send_json({
            "type": "ERROR",
            "message": "Recordings cannot be taken until this video is processed into keyposes.",
        })
        await websocket.close()
        return

    keyposes_dir = TEACHER_VIDEOS_DIR / video["keyposes_dir"]
    timestamp = datetime.now()
    rec_id = f"rec_{uuid4().hex}"
    formatted_date = timestamp.strftime("%b %d, %Y, %I:%M %p")

    raw_recording_path = RECORDINGS_DATA_DIR / f"{rec_id}_raw.mp4"
    final_recording_path = RECORDINGS_DATA_DIR / f"{rec_id}.mp4"

    def on_saved(final_path: Path, keypose_sync: List[Dict[str, Any]] | None = None, feedback=None) -> Dict[str, Any]:
        return register_new_recording(
            video_id=video["id"],
            video_name=video["name"],
            final_path=final_path,
            rec_id=rec_id,
            formatted_date=formatted_date,
            keypose_sync=keypose_sync,
            feedback=feedback,
        )

    if browser:
        with metadata_lock:
            busy = video_id in active_browser_videos or not get_video_by_id(video_id)
            if not busy:
                active_browser_videos.add(video_id)
        if busy:
            await websocket.send_json({"type": "ERROR", "message": "This video is unavailable or already in a training session."})
            await websocket.close()
            return

    def save_browser(path, sync, feedback):
        with metadata_lock:
            recording = on_saved(path, sync, feedback)
            try:
                status = process_recording(rec_id)["status"]
            except Exception:
                logging.exception("Could not queue the saved recording")
                status = "error"
                update_recording(rec_id, {"status": status, "error_message": "Could not queue 3D processing. Retry to continue."})
        return {**recording, "status": status}

    try:
        if browser:
            await run_browser_training(websocket, keyposes_dir, final_recording_path, save_browser)
        else:
            await run_web_training_session(
                websocket=websocket,
                keyposes_dir=keyposes_dir,
                raw_recording_path=raw_recording_path,
                final_recording_path=final_recording_path,
                on_recording_saved=on_saved,
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logging.exception("Training session failed")
        message = str(exc) if isinstance(exc, ValueError) else "Training ended before it could finish. Please try again."
        try:
            await websocket.send_json({"type": "ERROR", "message": message})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        if browser:
            with metadata_lock:
                active_browser_videos.discard(video_id)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Recordings & Motion Capture Pipeline Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/recordings")
def list_recordings():
    return load_metadata()


def store_upload(video, temp_path, final_path, strict):
    if video.size is not None and video.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload a video smaller than 100 MB.")
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        convert_to_browser_mp4(temp_path, final_path, strict=strict)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/recordings/upload")
async def upload_recording(
    video: UploadFile = File(...),
    video_id: str | None = Form(None),
    auto_process: bool = False,
):
    target_vid_id = (video_id or "misc_mocap").strip() or "misc_mocap"
    target_vid_name = "Test Motion Capture (Misc.)"

    if target_vid_id != "misc_mocap":
        with metadata_lock:
            videos = load_videos_locked()
            matched_video = next((v for v in videos if v["id"] == target_vid_id), None)
        if not matched_video:
            raise HTTPException(status_code=404, detail="Target teacher video not found")
        if not matched_video.get("is_processed", False):
            raise HTTPException(
                status_code=400,
                detail="Cannot take a recording for this video until it is processed into keyposes.",
            )
        target_vid_name = matched_video["name"]

    timestamp = datetime.now()
    rec_id = f"rec_{uuid4().hex}"
    formatted_date = timestamp.strftime("%b %d, %Y, %I:%M %p")

    raw_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    temp_path = RECORDINGS_DATA_DIR / f"{rec_id}_raw{raw_ext}"
    final_path = RECORDINGS_DATA_DIR / f"{rec_id}.mp4"

    await run_in_threadpool(store_upload, video, temp_path, final_path, auto_process)

    recording = register_new_recording(
        video_id=target_vid_id,
        video_name=target_vid_name,
        final_path=final_path,
        rec_id=rec_id,
        formatted_date=formatted_date,
    )
    if auto_process:
        process_recording(rec_id)
        recording["status"] = "processing"
    return recording


@app.post("/api/recordings/{rec_id}/process")
def process_recording(rec_id: str):
    with metadata_lock:
        item = get_recording_by_id(rec_id)
        if not item:
            raise HTTPException(status_code=404, detail="Recording not found")
        if item.get("status") == "processing" or (
            item.get("status") == "completed" and item.get("mocap_mirrored")
        ):
            return {"status": item["status"]}
        video_path = RECORDINGS_DATA_DIR / item["filename"]
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Source video file missing")
        output_dir = OUTPUTS_DIR / item["output_dir_name"]
        previous_output_dir = None
        if item.get("glb_filename"):
            # Publish the corrected rig under a new URL only after it is ready.
            previous_output_dir = output_dir
            output_dir = OUTPUTS_DIR / f"{video_path.stem}-mirrored-{uuid4().hex}"
        active_stages[rec_id] = "Queued"
        active_logs[rec_id] = []
        update_recording(rec_id, {"status": "processing", "error_message": None})
        jobs.submit(run_pipeline_worker, rec_id, video_path, output_dir, previous_output_dir)

    return {"message": "Processing started", "status": "processing"}


@app.get("/api/recordings/{rec_id}/status")
def get_recording_status(rec_id: str):
    item = get_recording_by_id(rec_id)
    if not item:
        raise HTTPException(status_code=404, detail="Recording not found")

    return {
        "id": rec_id,
        "status": item.get("status", "unprocessed"),
        "is_processed": item.get("is_processed", False),
        "stage": active_stages.get(rec_id, ""),
        "logs": active_logs.get(rec_id, []),
        "error_message": item.get("error_message"),
        "glb_url": f"/outputs/{item['output_dir_name']}/{item['glb_filename']}" if item.get("glb_filename") else None,
        "blend_url": f"/outputs/{item['output_dir_name']}/{item['blend_filename']}" if item.get("blend_filename") else None,
        "fbx_url": f"/outputs/{item['output_dir_name']}/{item['fbx_filename']}" if item.get("fbx_filename") else None,
    }


@app.get("/api/recordings/{rec_id}/feedback")
def get_recording_feedback(rec_id: str):
    if not get_recording_by_id(rec_id):
        raise HTTPException(status_code=404, detail="Recording not found")
    path = RECORDINGS_DATA_DIR / f"{rec_id}.feedback.json"
    return FileResponse(path, media_type="application/json") if path.is_file() else []


@app.get("/api/recordings/{rec_id}/keyposes")
def get_recording_keyposes(rec_id: str) -> list[float]:
    item = get_recording_by_id(rec_id)
    if not item:
        raise HTTPException(status_code=404, detail="Recording not found")

    matches = set()
    if item.get("video_id") != "misc_mocap":
        for pose in item.get("keypose_sync") or []:
            if not isinstance(pose, dict):
                continue
            timestamp = pose.get("mocap_match_time")
            if isinstance(timestamp, (int, float)) and math.isfinite(timestamp) and timestamp >= 0:
                matches.add(timestamp)
    if matches:
        return sorted(matches)

    if not item.get("glb_filename") or item.get("status") == "processing":
        return []
    landmarks = OUTPUTS_DIR / item["output_dir_name"] / "landmarks.json"
    try:
        return list(recording_keypose_times(landmarks, landmarks.stat().st_mtime_ns))
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        logging.warning("Could not read key poses for recording %s", rec_id, exc_info=True)
        return []


@app.delete("/api/recordings/{rec_id}")
def delete_recording(rec_id: str):
    with metadata_lock:
        item = get_recording_by_id(rec_id)
        if not item:
            raise HTTPException(status_code=404, detail="Recording not found")
        if item.get("status") == "processing":
            raise HTTPException(status_code=409, detail="Cannot delete a recording while it is processing")
        return delete_recording_media(item)


def delete_recording_media(item):
    rec_id = item["id"]

    video_path = RECORDINGS_DATA_DIR / item["filename"]
    video_path.unlink(missing_ok=True)
    (RECORDINGS_DATA_DIR / f"{rec_id}.feedback.json").unlink(missing_ok=True)

    out_dir = OUTPUTS_DIR / item["output_dir_name"]
    if out_dir.exists() and item["output_dir_name"] != "test":
        shutil.rmtree(out_dir, ignore_errors=True)

    active_logs.pop(rec_id, None)
    active_stages.pop(rec_id, None)

    with metadata_lock:
        items = [i for i in load_recordings_locked() if i["id"] != rec_id]
        renumber_recordings_locked(items)
        save_metadata_locked(items)

    return {"deleted": True, "id": rec_id, "recordings": items}


@app.get("/api/recordings/{rec_id}/side_video")
def serve_recording_side_video(rec_id: str):
    """Serve the side-by-side companion video for a processed motion capture recording.

    For any recording belonging to an uploaded teacher video (`video_id != 'misc_mocap'`),
    this ALWAYS returns the uploaded teacher video (`TEACHER_VIDEOS_DIR / parent_vid['filename']`),
    never the user's webcam recording.
    """
    rec = get_recording_by_id(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")

    vid_id = rec.get("video_id")
    if vid_id and vid_id != "misc_mocap":
        parent_vid = get_video_by_id(vid_id)
        if parent_vid and parent_vid.get("filename"):
            teacher_path = TEACHER_VIDEOS_DIR / parent_vid["filename"]
            if teacher_path.is_file():
                return FileResponse(teacher_path, media_type="video/mp4")

    rec_path = RECORDINGS_DATA_DIR / rec["filename"]
    if not rec_path.is_file():
        raise HTTPException(status_code=404, detail="Companion video not found")
    return FileResponse(rec_path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Static & Media File Serving
# ---------------------------------------------------------------------------
@app.get("/teacher_videos/{filename}")
def serve_teacher_video(filename: str):
    path = TEACHER_VIDEOS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Teacher video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/recordings/{filename}")
def serve_recording_file(filename: str):
    path = RECORDINGS_DATA_DIR / filename
    if path.suffix.lower() not in {".mp4", ".webm"} or not path.is_file():
        raise HTTPException(status_code=404, detail="Recording video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/outputs/{output_dir}/{filename}")
def serve_output_file(output_dir: str, filename: str):
    path = OUTPUTS_DIR / output_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    media_type = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/environment/dojo.glb")
def serve_dojo_environment():
    path = DOJO_GLB_PATH
    if not path.is_file():
        alt = WORKSPACE_DIR / "dojo.glb"
        if alt.is_file():
            path = alt
        else:
            raise HTTPException(status_code=404, detail="Dojo.glb not found in workspace root")
    return FileResponse(path, media_type="model/gltf-binary")


if PRODUCT_DIST is not None:
    app.mount("/assets", StaticFiles(directory=PRODUCT_DIST / "assets"), name="product-assets")
    app.mount("/media", StaticFiles(directory=PRODUCT_DIST / "media"), name="product-media")

    @app.get("/training", include_in_schema=False)
    @app.get("/analysis", include_in_schema=False)
    def serve_product():
        return FileResponse(PRODUCT_DIST / "index.html", media_type="text/html")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if PRODUCT_DIST is not None:
        return serve_product()
    return HTMLResponse("<h1>Frontend build not found. Run npm run build in final-web-app.</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn

    load_metadata()
    print("Starting MotionLearn Web App on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
