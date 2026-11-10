"""Regression tests for continuous recording and saved pose feedback."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import cv2
import numpy as np

from browser_training import BrowserTraining, FPS, Take


class PoseFeedbackTest(unittest.TestCase):
    def test_review_keeps_all_measured_joints_while_live_feedback_keeps_two(self):
        joints = ["L_armpit", "R_elbow", "L_knee", "hip_tilt"]
        session = BrowserTraining.__new__(BrowserTraining)
        session.pipeline = SimpleNamespace(
            mark_frame=lambda *args, **kwargs: (None, None, None, object()),
            human_analysis=lambda world: (dict(zip(joints, [40, 35, 30, 5])), None),
            joint_weights={},
            joint_bone=dict.fromkeys(joints),
        )
        session.targets = {"0": dict.fromkeys(joints, 0)}
        session.pose = 0
        session.state = "PAUSED"
        session.score = None
        session.hold = 0

        live, arrows, review = session.assess(None)
        self.assertEqual([item["id"] for item in live], joints[:2])
        self.assertEqual(arrows, [])
        self.assertEqual([item["id"] for item in review], joints)


class TrainingProgressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_poses_advance_without_intermediate_confirmation(self):
        session = BrowserTraining.__new__(BrowserTraining)
        session.ws = SimpleNamespace(send_json=AsyncMock())
        session.take = SimpleNamespace(frames=12, sync={}, feedback=[])
        session.pose = 0
        session.total = 2
        session.timestamps = {"0": 0.1, "1": 3.0}
        session.trajectories = {}
        session.state = "SEEKING"
        session.reset_scoring()

        self.assertFalse(await session.control({"type": "SEEKED"}))
        self.assertEqual(session.state, "PAUSED")
        self.assertFalse(await session.control({"type": "SKIP_POSE"}))
        self.assertEqual(session.pose, 1)
        self.assertEqual(session.state, "PLAYING")
        session.ws.send_json.assert_awaited_with({"action": "play_until", "target_time": 3.0, "state": "PLAYING"})
        self.assertFalse(await session.control({"type": "ARRIVED"}))
        self.assertEqual(session.state, "PAUSED")
        session.take.frames = 24
        self.assertTrue(await session.advance())
        self.assertEqual(session.take.sync, {
            0: {"arrive": 12, "match": 12},
            1: {"arrive": 24, "match": 24},
        })


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required to verify the final MP4")
class ContinuousRecordingTest(unittest.TestCase):
    def test_pose_and_feedback_times_use_the_continuous_recording(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            take = Take(root)
            for index in range(24):
                color = (220, 0, 0) if index < 12 else (0, 220, 0)
                frame = np.full((48, 64, 3), color, dtype=np.uint8)
                take.write(frame, round(index * 1000 / FPS))
            take.sync.update({0: {"arrive": 0, "match": 10}, 1: {"arrive": 12, "match": 22}})
            take.feedback.extend([
                (2, [{"id": "L_elbow", "degrees": 30}]),
                (10, []),
                (14, [{"id": "R_elbow", "degrees": 31}]),
            ])

            output = root / "saved.mp4"
            sync, feedback, duration = take.finish(output)
            self.assertEqual(duration, 24 / FPS)
            self.assertEqual([item["pose_id"] for item in sync], [0, 1])
            self.assertEqual(sync[1]["mocap_arrive_time"], 12 / FPS)
            self.assertEqual([item["joints"][0]["id"] for item in feedback], ["L_elbow", "R_elbow"])
            self.assertEqual(feedback[1]["start"], 14 / FPS)

            capture = cv2.VideoCapture(str(output))
            channels = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    channels.append(int(frame.mean(axis=(0, 1)).argmax()))
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), FPS)
            finally:
                capture.release()
            self.assertEqual(channels, [0] * 12 + [1] * 12)
