"""Regression test for checkpoint rewinds and the final playback timeline."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

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


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required to verify the final MP4")
class CheckpointRecordingTest(unittest.TestCase):
    def test_retake_keeps_prefix_and_removes_discarded_footage_and_feedback(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            take = Take(root)

            def record(color, joint, pose, epoch):
                frame = np.full((48, 64, 3), color, dtype=np.uint8)
                for index in range(12):
                    take.write(frame, epoch + round(index * 1000 / FPS))
                take.current.sync[pose] = {"arrive": 0, "match": 10}
                take.current.feedback.append((2, [{"id": joint, "degrees": 30}]))

            record((220, 0, 0), "L_elbow", 0, 0)
            take.new_segment(2)
            for attempt in range(2):
                record((0, 0, 220), "R_knee", 2, 1000 * (attempt + 1))
                self.assertEqual(take.rewind(previous=False), 2)
            record((0, 0, 220), "R_knee", 2, 3000)
            take.new_segment(4)
            record((0, 0, 220), "R_knee", 4, 4000)
            self.assertEqual(take.rewind(previous=True), 2)
            record((0, 220, 0), "R_elbow", 2, 5000)

            output = root / "saved.mp4"
            sync, feedback, duration = take.finish(output)
            self.assertEqual(duration, 24 / FPS)
            self.assertEqual([item["pose_id"] for item in sync], [0, 2])
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
