"""Browser camera training and clean video recording."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from fastapi import WebSocketDisconnect

from training_session import (
    JOINT_DEADZONE, REQUIRED_HOLD_FRAMES, SCORE_EMA_ALPHA,
    calculate_pose_match_score, update_hold_progress,
)
from utilities.dtw_scoring import score_transition
from utilities.mediapipeline import MedaiPipeline

FPS = 24
MAX_FRAME_BYTES = 512 * 1024


class Take:
    def __init__(self, directory: Path):
        self.path = directory / "capture.mp4"
        self.writer = None
        self.frame_size = None
        self.frames = 0
        self.sync = {}
        self.feedback = []

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def write(self, frame, timestamp):
        height, width = frame.shape[:2]
        if self.frame_size is None:
            self.frame_size = (width, height)
        if self.frame_size != (width, height):
            raise ValueError("The camera frame size changed during training.")
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.frame_size)
            if not self.writer.isOpened():
                raise RuntimeError("Could not start the clean recording.")
            self.epoch = timestamp
        elif timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 5000:
            raise ValueError("The camera stream was interrupted.")

        index = round((timestamp - self.epoch) * FPS / 1000)
        self.last_timestamp = timestamp
        if index < self.frames:
            return None
        while self.frames < index:
            self.writer.write(self.last_frame)
            self.frames += 1
        self.writer.write(frame)
        self.frames += 1
        self.last_frame = frame
        return index

    def finish(self, output):
        self.close()
        duration = self.frames / FPS
        if duration < 0.5:
            return None
        encoded = self.path.with_name("final.mp4")
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(self.path),
            "-vf", f"setpts=N/({FPS}*TB)", "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(encoded),
        ], check=True, timeout=max(180, 2 * duration))
        sync = [
            {
                "pose_id": pose,
                "mocap_arrive_time": entry["arrive"] / FPS,
                "mocap_match_time": entry.get("match", self.frames) / FPS,
            }
            for pose, entry in self.sync.items()
        ]
        feedback = []
        for position, (index, joints) in enumerate(self.feedback):
            if joints:
                following = self.feedback[position + 1][0] if position + 1 < len(self.feedback) else self.frames
                end = min(following, index + FPS // 4)
                if end > index:
                    feedback.append({"start": index / FPS, "end": end / FPS, "joints": joints})
        encoded.replace(output)
        return sync, feedback, duration


def decode_frame(payload):
    if len(payload) < 6 or len(payload) > MAX_FRAME_BYTES or payload[4:6] != b"\xff\xd8":
        raise ValueError("Invalid camera frame.")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8, offset=4), cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 1280 or any(value % 2 for value in frame.shape[:2]):
        raise ValueError("Unsupported camera frame dimensions.")
    return frame, int.from_bytes(payload[:4], "big")


class BrowserTraining:
    def __init__(self, websocket, keyposes_dir, take):
        self.ws = websocket
        self.take = take
        self.timestamps = self.read(keyposes_dir, "timestamps")
        self.poses = self.read(keyposes_dir, "keyposes")
        self.trajectories = self.read(keyposes_dir, "trajectories", optional=True)
        self.total = len(self.timestamps)
        if not self.total:
            raise ValueError("No teacher keyposes are available.")
        self.pipeline = MedaiPipeline(enable_ble=False, ema_alpha=0.65)
        self.targets = {key: self.pipeline.human_analysis(pose)[0] for key, pose in self.poses.items()}
        self.normalized = {key: self.pipeline.normalize_pose(pose)[0] for key, pose in self.poses.items()}
        self.pose = 0
        self.state = "SEEKING"
        self.reset_scoring()

    @staticmethod
    def read(directory, name, optional=False):
        path = directory / f"teacher_{name}.json"
        return {} if optional and not path.exists() else json.loads(path.read_text(encoding="utf-8"))

    def reset_scoring(self):
        self.hold = 0
        self.score = None
        self.dtw = None
        self.transition = []

    async def play_to_pose(self):
        self.state = "PLAYING"
        await self.ws.send_json({"action": "play_until", "target_time": self.timestamps[str(self.pose)], "state": self.state})

    async def advance(self):
        self.take.sync.setdefault(self.pose, {"arrive": self.take.frames})["match"] = self.take.frames
        self.take.feedback.append((self.take.frames, []))
        if self.pose + 1 == self.total:
            return True
        self.pose += 1
        self.reset_scoring()
        await self.play_to_pose()
        return False

    async def control(self, message):
        kind = message.get("type")
        if kind == "STOP_SESSION":
            return True
        if kind == "SEEKED" and self.state == "SEEKING":
            self.state = "PAUSED"
            await self.ws.send_json({"action": "pause", "state": self.state})
        elif kind == "ARRIVED" and self.state == "PLAYING":
            trajectory = self.trajectories.get(str(self.pose - 1))
            if trajectory and len(self.transition) >= 2:
                self.dtw = await asyncio.to_thread(score_transition, trajectory, self.transition)
            self.transition = []
            self.state = "PAUSED"
        elif kind == "SKIP_POSE" and self.state == "PAUSED":
            return await self.advance()
        return False

    def assess(self, frame):
        _, screen, _, world = self.pipeline.mark_frame(frame, draw_landmarks=False)
        if world is None:
            self.hold = 0
            self.score = None
            return [], [], []
        if self.state == "PLAYING":
            self.transition.append(self.pipeline.serialize_pose(world))
            return [], [], []

        key = str(self.pose)
        angles, _ = self.pipeline.human_analysis(world)
        score, errors = calculate_pose_match_score(self.targets[key], angles, self.pipeline.joint_weights, limit=None)
        if score is None or not math.isfinite(score):
            self.hold = 0
            self.score = None
            return [], [], []
        self.score = float(score if self.score is None else SCORE_EMA_ALPHA * score + (1 - SCORE_EMA_ALPHA) * self.score)
        self.hold, _ = update_hold_progress(self.score, self.hold)
        review_errors = [
            {"id": joint, "degrees": float(error)}
            for joint, error in errors if joint in self.pipeline.joint_bone
        ]
        mistakes = [
            {"id": joint, "degrees": round(float(error), 1)}
            for joint, error in errors[:2] if error >= JOINT_DEADZONE
        ]
        if not mistakes or screen is None:
            return mistakes, [], review_errors
        normalized, transform = self.pipeline.normalize_pose(world)
        screen_pose, screen_transform = self.pipeline.normalize_pose(screen)
        corrections = self.pipeline.euclidean_distance(
            teacher=self.normalized[key], student=normalized, student_params=transform,
            image=frame, render_pose=screen, render_student=screen_pose, render_params=screen_transform,
            allowed_landmarks=[self.pipeline.joint_bone[joint["id"]]["point"] for joint in mistakes], max_corrections=2,
        )
        height, width = frame.shape[:2]
        arrows = []
        for item in corrections:
            if item["start_point"] is not None and item["end_point"] is not None:
                x, y = item["start_point"]
                u, v = item["end_point"]
                arrows.append({"start": [float(x) / width, float(y) / height], "end": [float(u) / width, float(v) / height]})
        return mistakes, arrows, review_errors

    async def frame(self, payload):
        mistakes, arrows = [], []
        done = False
        if self.state != "SEEKING":
            frame, timestamp = await asyncio.to_thread(decode_frame, payload)
            index = await asyncio.to_thread(self.take.write, frame, timestamp)
            if index is not None:
                paused = self.state == "PAUSED"
                if paused:
                    self.take.sync.setdefault(self.pose, {"arrive": index})
                mistakes, arrows, review_errors = await asyncio.to_thread(self.assess, frame)
                if paused:
                    self.take.feedback.append((index, review_errors))
                if self.hold >= REQUIRED_HOLD_FRAMES:
                    done = await self.advance()
                    mistakes, arrows = [], []
        await self.ws.send_json({
            "type": "TRAINING_FEEDBACK", "state": self.state, "pose": self.pose + 1, "total": self.total,
            "hold": self.hold, "required_hold": REQUIRED_HOLD_FRAMES, "score": self.score,
            "dtw": self.dtw["overall_score"] if self.dtw else None, "joints": mistakes, "arrows": arrows,
        })
        return done

    async def capture(self):
        await self.ws.send_json({"action": "seek", "seek_time": self.timestamps["0"], "state": self.state})
        while True:
            message = await asyncio.wait_for(self.ws.receive(), timeout=15)
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if message.get("bytes") is not None:
                if await self.frame(message["bytes"]):
                    return
            elif await self.control(json.loads(message["text"])):
                return

    def timings(self, entries, duration):
        for index, entry in enumerate(entries):
            following = entries[index + 1] if index + 1 < len(entries) else None
            teacher_time = self.timestamps[str(entry["pose_id"])]
            entry.update(
                teacher_time=teacher_time,
                next_teacher_time=self.timestamps[str(following["pose_id"])] if following else teacher_time,
                next_mocap_arrive_time=following["mocap_arrive_time"] if following else duration,
            )
        return entries


async def run_browser_training(websocket, keyposes_dir, output, save):
    with TemporaryDirectory(prefix=".take-", dir=output.parent) as directory:
        take = Take(Path(directory))
        session = None
        saved = False
        try:
            session = await asyncio.to_thread(BrowserTraining, websocket, keyposes_dir, take)
            await session.capture()
            await websocket.send_json({"type": "SAVING"})
            result = await asyncio.to_thread(take.finish, output)
            if result is None:
                await websocket.send_json({"type": "SESSION_CANCELLED", "message": "The recording was too short to save."})
                return
            sync, feedback, duration = result
            recording = await asyncio.to_thread(save, output, session.timings(sync, duration), feedback)
            saved = True
            await websocket.send_json({"type": "SESSION_COMPLETE", "recording": recording})
        finally:
            take.close()
            if not saved:
                output.unlink(missing_ok=True)
            if session is not None:
                session.pipeline.landmarker.close()
