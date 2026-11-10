"""Interactive Copy-Training Session & Clean User Video Recorder.

Integrates the logic from `utilities/main_pipeline.py` and `youtube_extension`
directly with the FastAPI WebSocket and HTML5 Teacher Video Player, while
recording the clean (un-annotated) user webcam feed to an MP4 file for the
Motion Capture pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Callable, Dict

import cv2

from utilities.dtw_scoring import score_transition
from utilities.handrecognizer import HandRecognizer
from utilities.mediapipeline import MedaiPipeline
from utilities.video_mediapipeline import (
    human_analysis_segmentation,
    quat_to_dict,
    vect_to_dict,
)

MATCH_THRESHOLD = 16.0
NEAR_MATCH_THRESHOLD = 24.0
REQUIRED_HOLD_FRAMES = 8
JOINT_DEADZONE = 7.0
SCORE_EMA_ALPHA = 0.35
OPENCV_FRAME_FPS = 12
OPENCV_FRAME_INTERVAL = 1.0 / OPENCV_FRAME_FPS
OPENCV_FRAME_MAX_WIDTH = 640
OPENCV_FRAME_JPEG_QUALITY = 65


def encode_opencv_frame(frame) -> str | None:
    if frame is None:
        return None

    height, width = frame.shape[:2]
    frame_to_send = frame
    if width > OPENCV_FRAME_MAX_WIDTH:
        scale = OPENCV_FRAME_MAX_WIDTH / width
        frame_to_send = cv2.resize(
            frame,
            (OPENCV_FRAME_MAX_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        frame_to_send,
        [int(cv2.IMWRITE_JPEG_QUALITY), OPENCV_FRAME_JPEG_QUALITY],
    )
    if not success:
        return None

    return base64.b64encode(encoded.tobytes()).decode("ascii")


def calculate_pose_match_score(target_angles: dict, student_angles: dict, joint_weights: dict):
    weighted_error = 0.0
    total_weight = 0.0
    joint_errors = []

    for joint, target_angle in target_angles.items():
        student_angle = student_angles.get(joint)
        if target_angle is None or student_angle is None:
            continue

        raw_diff = abs(target_angle - student_angle)
        adjusted_diff = 0.0 if raw_diff < JOINT_DEADZONE else raw_diff
        weight = joint_weights.get(joint, 1.0)

        weighted_error += adjusted_diff * weight
        total_weight += weight
        joint_errors.append((joint, raw_diff))

    if total_weight == 0.0:
        return None, []

    joint_errors.sort(key=lambda item: item[1], reverse=True)
    return weighted_error / total_weight, joint_errors[:2]


def update_hold_progress(smoothed_score: float, match_hold_frames: int):
    if smoothed_score < MATCH_THRESHOLD:
        return match_hold_frames + 1, True
    if smoothed_score < NEAR_MATCH_THRESHOLD:
        return max(match_hold_frames - 1, 0), False
    return max(match_hold_frames - 3, 0), False


def finalize_clean_video(raw_path: Path, final_path: Path) -> bool:
    """Convert raw OpenCV MP4 into browser-friendly H.264 MP4."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return False

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        try:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(raw_path),
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
                "-an",
                str(final_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raw_path.unlink(missing_ok=True)
            return final_path.exists()
        except Exception:
            if not final_path.exists():
                raw_path.rename(final_path)
            return final_path.exists()
    else:
        if not final_path.exists():
            raw_path.rename(final_path)
        return final_path.exists()


def build_keypose_sync_for_recording(
    video_path: Path,
    keyposes_dir: Path,
    recorded_sync: Dict[int, Dict[str, float]] | list[Dict[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    """Build a complete per-keypose synchronization table mapping mocap time to teacher video time.

    Each returned item has:
      - pose_id: int (0 .. K-1)
      - teacher_time: float (timestamp of keypose in uploaded teacher video)
      - mocap_arrive_time: float (mocap time when teacher video arrives & pauses at teacher_time)
      - mocap_match_time: float (mocap time when user in recorded mocap matches the keypose)
      - next_teacher_time: float (timestamp of next keypose in teacher video)
      - next_mocap_arrive_time: float (mocap time when teacher video arrives at next_teacher_time)
    """
    timestamps_file = keyposes_dir / "teacher_timestamps.json"
    keyposes_file = keyposes_dir / "teacher_keyposes.json"
    if not timestamps_file.exists():
        return []

    try:
        raw_timestamps = json.loads(timestamps_file.read_text(encoding="utf-8"))
        timestamps = {int(k): float(v) for k, v in raw_timestamps.items()}
    except Exception:
        return []

    sorted_ids = sorted(timestamps.keys())
    total_poses = len(sorted_ids)
    if total_poses == 0:
        return []

    mocap_duration = 15.0
    fps = 30.0
    total_frames = 0
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0 and fps > 0:
                mocap_duration = max(1.0, total_frames / fps)
        cap.release()

    # Normalize recorded_sync into a dict keyed by int pose_id
    sync_by_id: Dict[int, Dict[str, Any]] = {}
    if isinstance(recorded_sync, dict):
        for k, v in recorded_sync.items():
            try:
                sync_by_id[int(k)] = dict(v)
            except Exception:
                pass
    elif isinstance(recorded_sync, list):
        for item in recorded_sync:
            if isinstance(item, dict) and "pose_id" in item:
                sync_by_id[int(item["pose_id"])] = dict(item)

    # Check if recorded_sync has keyposes recorded during the live session
    recorded_pids = [
        pid
        for pid in sorted_ids
        if pid in sync_by_id
        and sync_by_id[pid].get("mocap_arrive_time") is not None
        and sync_by_id[pid].get("mocap_match_time") is not None
    ]

    if len(recorded_pids) == total_poses:
        result: list[Dict[str, Any]] = []
        for idx, pid in enumerate(sorted_ids):
            entry = sync_by_id[pid]
            t_teacher = timestamps[pid]
            next_t_teacher = timestamps[sorted_ids[idx + 1]] if (idx + 1) < total_poses else t_teacher
            arrive_t = float(entry["mocap_arrive_time"])
            match_t = max(arrive_t + 0.1, float(entry["mocap_match_time"]))
            if (idx + 1) < total_poses:
                next_arrive_t = float(sync_by_id[sorted_ids[idx + 1]]["mocap_arrive_time"])
            else:
                next_arrive_t = max(match_t + 0.1, mocap_duration)
            result.append({
                "pose_id": pid,
                "teacher_time": round(t_teacher, 4),
                "mocap_arrive_time": round(arrive_t, 4),
                "mocap_match_time": round(min(match_t, next_arrive_t), 4),
                "next_teacher_time": round(next_t_teacher, 4),
                "next_mocap_arrive_time": round(next_arrive_t, 4),
            })
        return result

    # Otherwise, partition mocap_duration across all keyposes and detect best pose match per segment
    samples: list[tuple[float, dict | None]] = []
    target_angles_db: Dict[int, dict] = {}
    pipeline = None

    if keyposes_file.exists() and video_path and video_path.exists() and total_frames > 0:
        try:
            stored_poses = json.loads(keyposes_file.read_text(encoding="utf-8"))
            pipeline = MedaiPipeline(enable_ble=False, ema_alpha=0.65)
            for pid in sorted_ids:
                k_str = str(pid)
                if k_str in stored_poses:
                    angles, _ = pipeline.human_analysis(stored_poses[k_str])
                    target_angles_db[pid] = angles

            cap = cv2.VideoCapture(str(video_path))
            step = max(1, int(round(fps / 6.0)))  # ~6 samples/sec for fast analysis
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx % step == 0:
                    _, _, _, sw = pipeline.mark_frame(frame, draw_landmarks=False)
                    s_angles = pipeline.human_analysis(sw)[0] if sw is not None else None
                    samples.append((f_idx / fps, s_angles))
                f_idx += 1
            cap.release()
        except Exception as exc:
            print(f"[keypose_sync] Pose analysis fallback warning: {exc}")

    # Weight each keypose segment by its teacher transition + hold allowance
    weights: list[float] = []
    for idx, pid in enumerate(sorted_ids):
        t_cur = timestamps[pid]
        t_nxt = timestamps[sorted_ids[idx + 1]] if (idx + 1) < total_poses else t_cur
        trans = max(0.0, t_nxt - t_cur)
        weights.append(1.5 + trans)
    total_weight = sum(weights) or float(total_poses)

    result = []
    cursor = 0.0
    for idx, pid in enumerate(sorted_ids):
        seg_dur = mocap_duration * (weights[idx] / total_weight)
        arrive_t = cursor
        next_arrive_t = mocap_duration if (idx + 1) == total_poses else (cursor + seg_dur)
        t_teacher = timestamps[pid]
        next_t_teacher = timestamps[sorted_ids[idx + 1]] if (idx + 1) < total_poses else t_teacher

        # Default match time leaves time to hold pose and time to play transition to next keypose
        teacher_trans = max(0.0, next_t_teacher - t_teacher)
        play_window = min(seg_dur * 0.45, teacher_trans) if (idx + 1) < total_poses else (seg_dur * 0.15)
        default_match_t = max(arrive_t + seg_dur * 0.25, next_arrive_t - max(0.25, play_window))

        best_match_t = default_match_t
        if samples and pipeline and pid in target_angles_db:
            t_angles = target_angles_db[pid]
            search_start = arrive_t + seg_dur * 0.15
            search_end = next_arrive_t - max(0.2, play_window * 0.5)
            if search_end <= search_start:
                search_end = arrive_t + seg_dur * 0.75
            best_err = 9999.0
            for t_s, s_angles in samples:
                if search_start <= t_s <= search_end and s_angles is not None:
                    compared = sum(
                        1 for j, tv in t_angles.items() if tv is not None and s_angles.get(j) is not None
                    )
                    if compared >= 3:
                        err, _ = calculate_pose_match_score(t_angles, s_angles, pipeline.joint_weights)
                        if err is not None and err < best_err:
                            best_err = err
                            best_match_t = t_s

        result.append({
            "pose_id": pid,
            "teacher_time": round(t_teacher, 4),
            "mocap_arrive_time": round(arrive_t, 4),
            "mocap_match_time": round(best_match_t, 4),
            "next_teacher_time": round(next_t_teacher, 4),
            "next_mocap_arrive_time": round(next_arrive_t, 4),
        })
        cursor = next_arrive_t

    return result



async def run_web_training_session(
    websocket,
    video_info: Dict[str, Any],
    keyposes_dir: Path,
    raw_recording_path: Path,
    final_recording_path: Path,
    on_recording_saved: Callable[[Path], Dict[str, Any]],
    student_camera: int = 0,
) -> None:
    """Run the interactive copy-training session over FastAPI WebSocket.

    - Sends `seek`, `pause`, `play_until` commands to the web video player.
    - Streams `OPENCV_FRAME` (annotated with skeleton, arrows, and feedback)
      to the player's webcam overlay.
    - Writes the CLEAN, un-annotated `student_frame` to `raw_recording_path`
      so the saved user recording has no overlays and is ready for Motion Capture.
    """
    keyposes_file = keyposes_dir / "teacher_keyposes.json"
    timestamps_file = keyposes_dir / "teacher_timestamps.json"
    trajectories_file = keyposes_dir / "teacher_trajectories.json"

    if not keyposes_file.exists() or not timestamps_file.exists():
        await websocket.send_json({
            "type": "ERROR",
            "message": "Keyposes not found for this video. Please process the video first.",
        })
        return

    stored_poses = json.loads(keyposes_file.read_text(encoding="utf-8"))
    raw_timestamps = json.loads(timestamps_file.read_text(encoding="utf-8"))
    # Normalize keys to strings "0", "1", ...
    timestamps: Dict[str, float] = {str(k): float(v) for k, v in raw_timestamps.items()}
    total_poses = len(timestamps)

    teacher_trajectories = {}
    if trajectories_file.exists():
        try:
            teacher_trajectories = json.loads(trajectories_file.read_text(encoding="utf-8"))
        except Exception:
            teacher_trajectories = {}

    checkpoint_poses = int(video_info.get("checkpoint_poses") or max(1, total_poses))
    if checkpoint_poses <= 0:
        checkpoint_poses = max(1, total_poses)

    pipeline = MedaiPipeline(enable_ble=False, ema_alpha=0.65)
    try:
        hand_recognizer = HandRecognizer()
    except Exception as exc:
        print(f"HandRecognizer init warning: {exc}")
        hand_recognizer = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    segment_address = ("127.0.0.1", 5052)
    correction_address = ("127.0.0.1", 5053)

    target_angles_db = {}
    target_normalized_db = {}
    for key, pose_dict in stored_poses.items():
        angles, _ = pipeline.human_analysis(pose_dict)
        target_angles_db[str(key)] = angles
        normalized_pts, _ = pipeline.normalize_pose(pose_dict)
        target_normalized_db[str(key)] = normalized_pts

    cap_student = cv2.VideoCapture(student_camera)
    if not cap_student.isOpened():
        await websocket.send_json({
            "type": "ERROR",
            "message": "Could not open webcam. Make sure no other app or tab is using the camera.",
        })
        sock.close()
        return

    fps = cap_student.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 20.0
    width = int(cap_student.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap_student.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    raw_recording_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(raw_recording_path), fourcc, fps, (width, height))
    frames_written = 0

    pose_id = 0
    current_target_pose = str(pose_id)
    timestamp = timestamps.get(current_target_pose, 0.0)

    # Initial Seek & Pause on the frontend player
    await websocket.send_json({"action": "seek", "seek_time": timestamp})
    await websocket.send_json({"action": "pause"})
    state = "PAUSED"

    session_record = {}
    recorded_keypose_sync: Dict[int, Dict[str, float]] = {}
    current_pose_arrive_frame = 0
    student_transition_poses = []
    last_dtw_score = None
    match_hold_frames = 0
    score_ema = None
    last_overlay_frame_sent = 0.0
    checkpoint_waiting_paused = False
    should_stop = False

    def record_pose_match(matched_pose_id: int) -> None:
        recorded_keypose_sync[matched_pose_id] = {
            "pose_id": matched_pose_id,
            "teacher_time": float(timestamps.get(str(matched_pose_id), 0.0)),
            "mocap_arrive_time": round(current_pose_arrive_frame / fps, 4),
            "mocap_match_time": round(frames_written / fps, 4),
        }

    try:
        while not should_stop:
            # Non-blocking check for incoming WebSocket messages from browser UI
            while True:
                try:
                    msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.005)
                    data = json.loads(msg_text)
                    msg_type = data.get("type")

                    if msg_type == "STOP_SESSION":
                        should_stop = True
                        break
                    elif msg_type == "ARRIVED" and state == "PLAYING":
                        transition_key = str(pose_id - 1)
                        teacher_traj = teacher_trajectories.get(transition_key, [])
                        if teacher_traj and len(student_transition_poses) >= 2:
                            last_dtw_score = score_transition(teacher_traj, student_transition_poses)
                        else:
                            last_dtw_score = None
                        student_transition_poses = []
                        current_pose_arrive_frame = frames_written
                        state = "PAUSED"
                    elif msg_type == "SKIP_POSE" and state == "PAUSED":
                        next_timestamp = timestamps.get(str(pose_id + 1))
                        if ((pose_id + 1) % checkpoint_poses) == 0 and next_timestamp is not None:
                            state = "CHECKPOINT"
                            checkpoint_waiting_paused = False
                            match_hold_frames = 0
                            score_ema = None
                        elif next_timestamp is not None:
                            record_pose_match(pose_id)
                            await websocket.send_json({"action": "play_until", "target_time": next_timestamp})
                            state = "PLAYING"
                            pose_id += 1
                            current_target_pose = str(pose_id)
                            timestamp = next_timestamp
                            match_hold_frames = 0
                            score_ema = None
                        else:
                            record_pose_match(pose_id)
                            should_stop = True
                            break
                    elif msg_type == "CHECKPOINT_ACTION" and state == "CHECKPOINT":
                        ui_action = data.get("action")
                        if ui_action == "Thumb_Up":
                            next_timestamp = timestamps.get(str(pose_id + 1))
                            record_pose_match(pose_id)
                            if next_timestamp is not None:
                                await websocket.send_json({"action": "play_until", "target_time": next_timestamp})
                                state = "PLAYING"
                                checkpoint_waiting_paused = False
                                pose_id += 1
                                current_target_pose = str(pose_id)
                                timestamp = next_timestamp
                                match_hold_frames = 0
                                score_ema = None
                            else:
                                should_stop = True
                                break
                        elif ui_action == "Thumb_Down":
                            pose_id = max(0, pose_id - checkpoint_poses + 1)
                            next_timestamp = timestamps.get(str(pose_id))
                            if next_timestamp is not None:
                                await websocket.send_json({"action": "seek", "seek_time": next_timestamp})
                                await websocket.send_json({"action": "pause"})
                                current_target_pose = str(pose_id)
                                timestamp = next_timestamp
                                current_pose_arrive_frame = frames_written
                                match_hold_frames = 0
                                score_ema = None
                                student_transition_poses = []
                                replay_target = timestamps.get(str(pose_id + 1))
                                if replay_target is not None:
                                    record_pose_match(pose_id)
                                    await websocket.send_json({"action": "play_until", "target_time": replay_target})
                                    state = "PLAYING"
                                    checkpoint_waiting_paused = False
                                    pose_id += 1
                                    current_target_pose = str(pose_id)
                                    timestamp = replay_target
                                else:
                                    state = "PAUSED"
                                    checkpoint_waiting_paused = False
                except asyncio.TimeoutError:
                    break

            if should_stop:
                break

            ret, student_frame = cap_student.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # 1. Write the CLEAN raw user frame (no overlays!) to the recording video
            video_writer.write(student_frame)
            frames_written += 1

            # 2. Process frame for pose matching and overlay visualization
            display_student = student_frame.copy()
            action = None
            worst_joint_names = []

            if state in ("PAUSED", "CHECKPOINT"):
                _, pose, world_pose, smoothed_world = pipeline.mark_frame(display_student, draw_landmarks=False)
                if hand_recognizer is not None:
                    try:
                        action = hand_recognizer.mark_frame(student_frame, int(time.time() * 1000))
                    except Exception:
                        action = None

                if world_pose is not None and smoothed_world is not None:
                    try:
                        segments = human_analysis_segmentation(world_pose)
                        segment_payload = {
                            "segments": {
                                k: quat_to_dict(v) if k == "head_tilt" else vect_to_dict(v)
                                for k, v in segments.items()
                                if v is not None
                            }
                        }
                        sock.sendto(json.dumps(segment_payload).encode(), segment_address)
                    except Exception:
                        pass

                    student_angles, _ = pipeline.human_analysis(smoothed_world)
                    student_normalized, student_transform = pipeline.normalize_pose(smoothed_world)
                    student_screen_normalized, student_screen_transform = pipeline.normalize_pose(pose)

                    target_angles = target_angles_db.get(current_target_pose)
                    target_normalized = target_normalized_db.get(current_target_pose)

                    if target_angles and state == "PAUSED":
                        average_score, worst_joints = calculate_pose_match_score(
                            target_angles,
                            student_angles,
                            pipeline.joint_weights,
                        )
                        worst_joint_names = [joint for joint, _ in worst_joints]

                        if average_score is not None:
                            if score_ema is None:
                                score_ema = average_score
                            else:
                                score_ema = (
                                    SCORE_EMA_ALPHA * average_score
                                    + (1.0 - SCORE_EMA_ALPHA) * score_ema
                                )

                            correction_landmarks = [
                                pipeline.joint_bone[joint]["point"]
                                for joint, _ in worst_joints
                                if joint in pipeline.joint_bone
                            ]
                            corrections = pipeline.euclidean_distance(
                                teacher=target_normalized,
                                student=student_normalized,
                                student_params=student_transform,
                                image=display_student,
                                render_pose=pose,
                                render_student=student_screen_normalized,
                                render_params=student_screen_transform,
                                allowed_landmarks=correction_landmarks,
                                max_corrections=2,
                            )
                            display_student = pipeline.draw_arrow(display_student, corrections)

                            correction_payload = pipeline.corrections_to_unity_format(corrections)
                            if any(v is not None for v in correction_payload["corrections"].values()):
                                try:
                                    sock.sendto(json.dumps(correction_payload).encode(), correction_address)
                                except Exception:
                                    pass

                            match_hold_frames, is_match = update_hold_progress(score_ema, match_hold_frames)
                            if is_match:
                                cv2.putText(
                                    display_student,
                                    f"HOLD IT! ({match_hold_frames}/{REQUIRED_HOLD_FRAMES}) Error: {score_ema:.1f}",
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )

                                if match_hold_frames >= REQUIRED_HOLD_FRAMES:
                                    session_record[current_target_pose] = {
                                        "score": average_score,
                                        "worst_joints": worst_joint_names,
                                        "dtw_score": last_dtw_score["overall_score"] if last_dtw_score else None,
                                    }
                                    student_transition_poses = []
                                    next_timestamp = timestamps.get(str(pose_id + 1))

                                    if ((pose_id + 1) % checkpoint_poses) == 0 and next_timestamp is not None:
                                        state = "CHECKPOINT"
                                        checkpoint_waiting_paused = False
                                        match_hold_frames = 0
                                        score_ema = None
                                        action = None
                                    elif next_timestamp is not None:
                                        record_pose_match(pose_id)
                                        await websocket.send_json({"action": "play_until", "target_time": next_timestamp})
                                        state = "PLAYING"
                                        pose_id += 1
                                        current_target_pose = str(pose_id)
                                        timestamp = next_timestamp
                                        match_hold_frames = 0
                                        score_ema = None
                                    else:
                                        # Reached final keypose!
                                        record_pose_match(pose_id)
                                        should_stop = True
                                        break
                            else:
                                cv2.putText(
                                    display_student,
                                    f"Match pose #{pose_id + 1}/{total_poses} (Err: {score_ema:.1f})",
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                if worst_joint_names:
                                    cv2.putText(
                                        display_student,
                                        f"Adjust: {', '.join(worst_joint_names)}",
                                        (20, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

            if state == "CHECKPOINT":
                if not checkpoint_waiting_paused:
                    await websocket.send_json({"action": "pause"})
                    checkpoint_waiting_paused = True
                cv2.putText(
                    display_student,
                    "CHECKPOINT: Thumbs Up to continue / Thumbs Down to retry",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if action == "Thumb_Up":
                    next_timestamp = timestamps.get(str(pose_id + 1))
                    record_pose_match(pose_id)
                    if next_timestamp is not None:
                        await websocket.send_json({"action": "play_until", "target_time": next_timestamp})
                        state = "PLAYING"
                        checkpoint_waiting_paused = False
                        pose_id += 1
                        current_target_pose = str(pose_id)
                        timestamp = next_timestamp
                        match_hold_frames = 0
                        score_ema = None
                        action = None
                    else:
                        should_stop = True
                        break
                elif action == "Thumb_Down":
                    pose_id = max(0, pose_id - checkpoint_poses + 1)
                    next_timestamp = timestamps.get(str(pose_id))
                    if next_timestamp is not None:
                        await websocket.send_json({"action": "seek", "seek_time": next_timestamp})
                        await websocket.send_json({"action": "pause"})
                        current_target_pose = str(pose_id)
                        timestamp = next_timestamp
                        current_pose_arrive_frame = frames_written
                        match_hold_frames = 0
                        score_ema = None
                        student_transition_poses = []
                        action = None
                        replay_target = timestamps.get(str(pose_id + 1))
                        if replay_target is not None:
                            record_pose_match(pose_id)
                            await websocket.send_json({"action": "play_until", "target_time": replay_target})
                            state = "PLAYING"
                            checkpoint_waiting_paused = False
                            pose_id += 1
                            current_target_pose = str(pose_id)
                            timestamp = replay_target
                        else:
                            state = "PAUSED"
                            checkpoint_waiting_paused = False

            elif state == "PLAYING":
                _, pose, world_pose, smoothed_world = pipeline.mark_frame(display_student, draw_landmarks=False)
                if smoothed_world is not None:
                    student_transition_poses.append(pipeline.serialize_pose(smoothed_world))
                cv2.putText(
                    display_student,
                    f"Follow the movement -> Pose #{pose_id + 1}/{total_poses}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            now = time.monotonic()
            if now - last_overlay_frame_sent >= OPENCV_FRAME_INTERVAL:
                encoded = encode_opencv_frame(display_student)
                if encoded:
                    await websocket.send_json({
                        "type": "OPENCV_FRAME",
                        "image": encoded,
                        "state": state,
                        "pose_id": pose_id,
                        "total_poses": total_poses,
                        "score_ema": round(score_ema, 1) if score_ema is not None else None,
                        "hold_frames": match_hold_frames,
                        "required_hold_frames": REQUIRED_HOLD_FRAMES,
                        "worst_joints": worst_joint_names,
                        "dtw_score": round(last_dtw_score["overall_score"], 1) if last_dtw_score else None,
                    })
                last_overlay_frame_sent = now

            await asyncio.sleep(0.005)

    except Exception as exc:
        print(f"[training_session] ended with exception: {exc}")
    finally:
        cap_student.release()
        video_writer.release()
        sock.close()

    # Finalize and register the clean user recording
    if frames_written > 5 and finalize_clean_video(raw_recording_path, final_recording_path):
        keypose_sync = build_keypose_sync_for_recording(
            video_path=final_recording_path,
            keyposes_dir=keyposes_dir,
            recorded_sync=recorded_keypose_sync,
        )
        new_rec = on_recording_saved(final_recording_path, keypose_sync)
        try:
            await websocket.send_json({
                "type": "SESSION_COMPLETE",
                "recording": new_rec,
            })
        except Exception:
            pass
    else:
        raw_recording_path.unlink(missing_ok=True)
        try:
            await websocket.send_json({
                "type": "SESSION_CANCELLED",
                "message": "Recording was too short or empty.",
            })
        except Exception:
            pass

