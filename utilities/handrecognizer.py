import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from pathlib import Path


GESTURE_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "youtube_extension"
    / "gesture_recognizer.task"
)


class HandRecognizer():
    def __init__(self) -> None:
        self.landmarker = self._initialize()

    def _initialize(self):
        if not GESTURE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Gesture recognizer model not found: {GESTURE_MODEL_PATH}")

        base_options = python.BaseOptions(model_asset_path=str(GESTURE_MODEL_PATH))
        options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO)
        recognizer = vision.GestureRecognizer.create_from_options(options)
        return recognizer

    def mark_frame(self, frame, timestamp_ms):
        image_3_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_3_channel)
        detection_result = self.landmarker.recognize_for_video(mp_image, timestamp_ms) # tracks landmarks across time
        if detection_result.gestures:
            action = detection_result.gestures[0][0].category_name
            return action
        return None
