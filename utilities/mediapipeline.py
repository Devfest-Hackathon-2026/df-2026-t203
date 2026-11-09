import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import numpy as np
try:
    from mediapipe.tasks.python.vision import drawing_utils
    from mediapipe.tasks.python.vision import drawing_styles
    _HAS_TASKS_DRAWING = True
except ImportError:
    from mediapipe.python.solutions import drawing_utils
    from mediapipe.python.solutions import drawing_styles
    from mediapipe.framework.formats import landmark_pb2
    _HAS_TASKS_DRAWING = False
import cv2
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

#---- esp32 comms

import logging
import asyncio
from dotenv import load_dotenv
import os
from bleak import BleakScanner, BleakClient
import threading
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
model_path = str(_REPO_ROOT / "pose_landmarker_full.task")
keyposes_path = str(_REPO_ROOT / "keyposes")
keypose_meta_path = str(_REPO_ROOT / "keyposes" / "poses.json")


    #relevant landmarks
LM ={
  "nose":0,
  "l_shoulder": 11,"r_shoulder": 12,
  "l_elbow": 13,"r_elbow": 14,
  "l_wrist": 15, "r_wrist": 16,
  "l_hip": 23, "r_hip": 24,
  "l_knee": 25, "r_knee": 26,
  "l_ankle": 27, "r_ankle": 28,
}

JOINT_BONE={ #which bone to target for that joint
    "R_armpit":   {"point": 14, "label": "right elbow"},
    "R_elbow":    {"point": 16, "label": "right forearm"},
    "L_armpit":   {"point": 13, "label": "left elbow"},
    "L_elbow":    {"point": 15, "label": "left forearm"},
    "chest_tilt": {"point": 12, "label": "Chest"},
    "hip_tilt":   {"point": 24, "label": "hips"},
    "R_pelvis":   {"point": 26, "label": "right knee"},
    "L_pelvis":   {"point": 25, "label": "left knee"},
    "R_knee":     {"point": 28, "label": "right ankle"},
    "L_knee":     {"point": 27, "label": "left ankle"},
}

JOINT_WEIGHTS = { #weighting for how important this stuff is
    "R_armpit": 1.0, "L_armpit": 1.0,
    "R_elbow":  0.8, "L_elbow":  0.8,
    "chest_tilt": 1.2, "hip_tilt": 1.2,
    "R_pelvis": 1.0, "L_pelvis": 1.0,
    "R_knee":   0.9, "L_knee":   0.9,
}

ANGLE_THRESHOLD = 12
VISIBILITY_THRESHOLD = 0.5 #must be above this value to be considered

load_dotenv()

CHARACTERISTIC_UUID = os.getenv("CHARACTERISTIC_UUID")
SERVICE_UUID = os.getenv("SERVICE_UUID")
ESP_DEVICE = None

class ESPTransmit:
  def __init__(self, target_device="NauESP32", char_uuid="CHARACTERISTIC_UUID"):
    self.target_device = target_device
    self.char_uuid = char_uuid
    self.device = None
    self.data_queue = queue.Queue(maxsize=1)
    self._running = False
    self._thread = None
  
  def start(self):
        """Starts the background BLE thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

  def stop(self):
      """Cleanly shuts down the thread."""
      self._running = False
      if self._thread is not None:
          self._thread.join()

  def send_data(self, payload: str):
      """Synchronous method called by OpenCV to safely pass data to the BLE thread."""
      if not self.data_queue.full():
          self.data_queue.put(payload)
      else:
          # If the queue is full (BLE is lagging), we drop the oldest frame 
          # to ensure the wearable always gets the freshest pose data.
          try:
              self.data_queue.get_nowait()
              self.data_queue.put(payload)
          except queue.Empty:
              pass
  
  def _run_async_loop(self):
        """Wrapper to run the async BLE loop inside our thread."""
        asyncio.run(self._ble_task())

  async def _find_device(self) -> bool:
      logger.info(f"Scanning for ESP devices matching {self.target_device}...")
      try:
          devices = await BleakScanner.discover(timeout=5.0)
          for device in devices:
              if device.name and device.name.lower() == self.target_device.lower():
                  self.device = device
                  logger.info(f"Found device: {self.device.name} - {self.device.address}")
                  return True
          logger.warning("No ESP device found")
          return False
      except Exception as e:
          logger.error(f"Error in BLE Scanning: {e}")
          return False

  async def _ble_task(self):
      found = await self._find_device()
      if not found or self.device is None:
          return

      logger.info(f"Connecting to {self.device.name}")
      try:
          async with BleakClient(self.device.address) as client:
              if client.is_connected:
                  logger.info("Connected successfully! Starting transmission loop...")
                  
                  # --- CONTINUOUS CONTROL LOOP ---
                  while self._running:
                      try:
                          # Non-blocking check for new data from OpenCV
                          payload = self.data_queue.get_nowait()
                          data = payload.encode('utf-8')
                          
                          await client.write_gatt_char(self.char_uuid, data, response=False)
                          # logger.info(f"Sent: {payload}")
                          
                      except queue.Empty:
                          # No new data, just sleep briefly to prevent CPU hogging
                          await asyncio.sleep(0.05)
                      except Exception as e:
                          logger.error(f"Error writing to BLE: {e}")
                          break
              else:
                  logger.warning("Failed connecting")
      except Exception as e:
          logger.error(f"Error connecting to client: {e}")

class EMAFilter:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.previous_pose = {}

    def process(self, current_pose_dict):
        if not self.previous_pose:
            self.previous_pose = current_pose_dict.copy()
            return current_pose_dict

        smoothed_pose = {}
        for k, v in current_pose_dict.items():
            if k in self.previous_pose:
                smoothed_pose[k] = (self.alpha * v) + ((1.0 - self.alpha) * self.previous_pose[k])
            else:
                smoothed_pose[k] = v
                
        self.previous_pose = smoothed_pose.copy()
        return smoothed_pose

class human():
  def __init__(self,angles,points) -> None:
    self.angles: dict = angles
    self.points: dict = points #normalized coordinates

class MedaiPipeline():
  
  def __init__(self, enable_ble=True, ema_alpha=0.5) -> None:
    self.landmarker = self._initialize()
    self.angle_threshold = ANGLE_THRESHOLD
    self.visibility_threshold = VISIBILITY_THRESHOLD
    self.lm = LM
    self.joint_bone = JOINT_BONE
    self.joint_weights = JOINT_WEIGHTS
    self.pose_filter = EMAFilter(alpha=ema_alpha)
    self.world_pose_filter = EMAFilter(alpha=ema_alpha)
    if enable_ble:
      self.ble_transmit = ESPTransmit(target_device="NauESP32", char_uuid=CHARACTERISTIC_UUID) #type:ignore
      self.ble_transmit.start()
    else:
      self.ble_transmit = None



  def _initialize(self):
    """
    Create the thing that creates the path
    
    :param self: Description
    """
    # setup
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    landmarker=vision.PoseLandmarker.create_from_options(options)
    return landmarker
    
  def draw_landmarks_on_image(self, rgb_image, detection_result):
      """
      Draw the landmarks on an image
      
      :param self: Description
      :param rgb_image: Description
      :param detection_result: Description
      """

      pose_landmarks_list = detection_result.pose_landmarks
      annotated_image = np.copy(rgb_image)

      pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
      pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

      for pose_landmarks in pose_landmarks_list:
        if _HAS_TASKS_DRAWING:
          drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=pose_landmarks,
              connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
              landmark_drawing_spec=pose_landmark_style,
              connection_drawing_spec=pose_connection_style)
        else:
          proto_list = landmark_pb2.NormalizedLandmarkList()
          proto_list.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
          ])
          drawing_utils.draw_landmarks(
              image=annotated_image,
              landmark_list=proto_list,
              connections=mp.solutions.pose.POSE_CONNECTIONS,
              landmark_drawing_spec=pose_landmark_style,
              connection_drawing_spec=pose_connection_style)

      return annotated_image
  
  def mark_image(self,file_name:str):
    """
    Create the landmarks
    
    :param self: Description
    :param file_name: Description
    :type file_name: str
    """

    mp_image=mp.Image.create_from_file(file_name=file_name)
    raw_image= mp_image.numpy_view()
    image_3_channel = cv2.cvtColor(raw_image,cv2.COLOR_RGBA2RGB)
    pose_landmarker_result = self.landmarker.detect(mp_image)
    annotated_image = self.draw_landmarks_on_image(image_3_channel, pose_landmarker_result)
    pose=pose_landmarker_result.pose_landmarks[0]
    image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return image, pose
  
  def human_analysis(self,smoothed_pose:dict):
    """
    Takes landmarks and generates angles for the body by making body into vectors and finding angles between them
    """
    #geometry helping:
    def vec(pose_dict,a,b):
      """Vector from a to b"""
      return pose_dict[b]-pose_dict[a]

    def find_angle(v1,v2):
      """
      finds angles between two vectors
      
      :param v1: first vector
      :param v2: second vector
      """
      v1 = v1/np.linalg.norm(v1)
      v2 = v2/np.linalg.norm(v2)

      # for quaternions
      # rotation= R.align_vectors([v1], [v2])[0]
      # quat = rotation.as_quat()

      return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    def signed_angle(v1,v2):
      """angle with direction, positive is counter-clockwise"""
      angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
      return float(np.degrees(angle))

    def normalize_orientation(pose_dict:dict):
      """Rotate everything so that we make hip-> shoulder vertical to ignore rotations and things"""
      mid_hip = (pose_dict[LM["r_hip"]] + pose_dict[LM["l_hip"]])/2
      mid_shoulder = (pose_dict[LM["r_shoulder"]] + pose_dict[LM["l_shoulder"]])/2
      spine = mid_shoulder - mid_hip
      spine_angle = np.arctan2(spine[1], spine[0])
      target_angle = np.pi/2
      rotation_needed = target_angle-spine_angle
      cos_a, sin_a = np.cos(rotation_needed),np.sin(rotation_needed)
      
      rot = np.array([ 
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
      ])
      centered = {k: v - mid_hip for k,v in pose_dict.items()}
      rotated = {k: rot @ v for k,v in centered.items()}
      return rotated

    values = {int(k): np.array(v[:3]) for k, v in smoothed_pose.items()}
    visibility = {int(k): v[3] for k, v in smoothed_pose.items()}
    
    def angle_if_vis(idx_list,v1,v2):
      """Return the angle of the body part if visible or not"""
      if all(visibility.get(i,1.0) >= VISIBILITY_THRESHOLD for i in idx_list):
        return find_angle(v1,v2)
      return None

    #------------
    n = normalize_orientation(pose_dict=values) #normalize the orientation

    body_angles = {
        "R_armpit":  angle_if_vis([12, 14, 24], vec(n,12,14), vec(n,12,24)),
        "R_elbow":   angle_if_vis([12, 14, 16], vec(n,14,16), vec(n,14,12)),
        "L_armpit":  angle_if_vis([11, 13, 23], vec(n,11,13), vec(n,11,23)),
        "L_elbow":   angle_if_vis([11, 13, 15], vec(n,13,15), vec(n,13,11)),
        "chest_tilt":  angle_if_vis([11, 12], vec(n,12,11), np.array([1,0,0])),
        "hip_tilt":    angle_if_vis([23, 24], vec(n,24,23), np.array([1,0,0])),
        "R_pelvis":  angle_if_vis([24, 26], vec(n,24,23), vec(n,24,26)),
        "L_pelvis":  angle_if_vis([23, 25], vec(n,23,24), vec(n,23,25)),
        "R_knee":    angle_if_vis([24, 26, 28], vec(n,26,24), vec(n,26,28)),
        "L_knee":    angle_if_vis([23, 25, 27], vec(n,25,23), vec(n,25,27)),
    }

    return body_angles,values

  def _hint_location(self,joint,t_val,s_val):
      """TO give the hints on what the angle is"""
      diff = t_val - s_val
      label = JOINT_BONE[joint]["label"]
      if "elbow" in joint.lower() or "knee" in joint.lower():
        return f"{label} : {'extend more' if diff > 0 else 'bend more'} ({diff:+.1f})"
      elif "armpit" in joint.lower() or "pelvis" in joint.lower():
        return f"{label}: {'raise' if diff < 0 else 'lower'} ({diff:+.1f})"
      else:
        return f"{label}: adjust by {diff:+.1f}"
      
  def difference(self,teacher:human,student:human, image, ax=None, threshold = ANGLE_THRESHOLD):
    """Compare the two angles, then annotate for correction"""
    corrections = []
    img_height, img_width = image.shape[:2]
    common = set(teacher.angles) & set(student.angles) #check which common joints they have

    for j in common:
      t_val = teacher.angles[j]
      s_val = student.angles[j]
      weight = JOINT_WEIGHTS.get(j,1.0) #get the weighting for importance

      final_threshold = threshold/weight
      if t_val is None or s_val is None:
        continue

      if abs(t_val-s_val) > final_threshold:
        bone = JOINT_BONE[j]
        point_indx = bone["point"]
        hint = self._hint_location(j,t_val,s_val)

        px_stu = int(float(student.points[point_indx][0]) * img_width)
        py_stu = int(float(student.points[point_indx][1]) * img_height)

        px_teach = int(float(teacher.points[point_indx][0]) * img_width)
        py_teach = int(float(teacher.points[point_indx][1]) * img_height)

        corrections.append({"hint":hint,"points":student.points[point_indx], "start_point":(px_stu, py_stu), "end_point": (px_teach, py_teach)})

        # px_teach = float(teacher.points[point_indx][0]) * img_width
        # py_teach = float(teacher.points[point_indx][1]) * img_height

        # ax.annotate(
        #   "",
        #   xy=(px,py),
        #   xytext=(px_teach,py_teach),
        #   arrowprops=dict(arrowstyle="->",color="blue")
        # )
        if ax is not None:
          ax.annotate(
            hint,
            xy=(px_stu,py_stu),
            xytext = (px_stu+40,py_stu+40),
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
          )

    return corrections

  def normalize(self,pose:dict):
    """normalize the pose so that everything is the same
    
    midpoint of hip -> midpoint to chest = 1
    divide everything by that
    
    """
    def get_value(indx):
      """I dont remember if i kept it as an integer or a string, so im using this"""
      if indx in pose:
        return np.array(pose[indx][:3])
      if str(indx) in pose:
        return np.array(pose[str(indx)][:3])
      return np.zeros(3)

    #chest
    chest = (get_value(12) + get_value(11))/2
    hip = (get_value(24) + get_value(23))/2
    
    
    normalizer = np.linalg.norm(chest - hip)

    if normalizer == 0:
      return pose

    for k,v in pose.items():
      pose[k] = [
        (v[0] - hip[0])/normalizer,
        (v[1] - hip[1])/normalizer,
        (v[2] - hip[2])/normalizer,
        v[3]
      ]
    return pose

  def compare_vectors(self,v1,v2):
    """
    dot product every vector, check score for every item, then find thing with lowest score and thats the one closest
    """
    x = abs(v1[0]-v2[0])
    y = abs(v1[1]-v2[1])
    z = abs(v1[2]-v2[2])

    p1 = np.array(v1[:3])
    p2 = np.array(v2[:3])

    return np.linalg.norm(p1-p2) #distance between two vectors
  
#----- functions for arrow
  
  def normalize_pose(self, pose_data):
    """
    For Euclidean distance between student and teacher pose
    Translate to origin, rotate spine to vertical and scale by torso length.
    torso length is between midpoint of shoulder and midpoint of hip
    """
    np_pose = {}
        
    iterable = pose_data.items() if isinstance(pose_data, dict) else enumerate(pose_data)
    
    for k, v in iterable:
        int_k = int(k) # <--- FIX: Force the key to be an integer here
        
        if hasattr(v, 'x'):
            np_pose[int_k] = np.array([v.x, v.y, getattr(v, 'z', 0.0)])
        else:
            np_pose[int_k] = np.array(v[:3])

    # Now this will work properly because the keys are guaranteed to be integers
    mid_hip = (np_pose[LM["r_hip"]] + np_pose[LM["l_hip"]]) / 2
    mid_shoulder = (np_pose[LM["r_shoulder"]] + np_pose[LM["l_shoulder"]]) / 2
    spine = mid_shoulder - mid_hip
    spine_angle = np.arctan2(spine[1], spine[0])
    target_angle = np.pi/2
    rotation_needed = target_angle - spine_angle
    cos_a, sin_a = np.cos(rotation_needed), np.sin(rotation_needed)
    
    rot = np.array([ 
      [cos_a, -sin_a, 0],
      [sin_a,  cos_a, 0],
      [0,      0,     1]
    ])
    centered = {k: v - mid_hip for k, v in np_pose.items()}
    rotated = {k: rot @ v for k, v in centered.items()}

    torso_length = np.linalg.norm(spine)
    
    # Protect against division by zero just in case
    if torso_length == 0:
        torso_length = 1.0 
        
    normalized_pose = {k: v/torso_length for k, v in rotated.items()}

    transform_params = {
      "mid_hip": mid_hip,
      "rot": rot,
      "torso_length": torso_length
    }

    return normalized_pose, transform_params
  
  def euclidean_distance(
      self,
      teacher: dict,
      student: dict,
      student_params: dict,
      image,
      ax=None,
      threshold=0.3,
      render_pose=None,
      render_student=None,
      render_params=None,
      allowed_landmarks=None,
      max_corrections=2,
      min_arrow_px=15.0,
      max_arrow_px=120.0,
  ):
    """
    Build correction vectors in body space, with optional 2D preview arrows.
    """
    corrections = []
    img_height, img_width = image.shape[:2]
    
    relevant_landmarks = set(self.lm.values())
    if allowed_landmarks is not None:
        relevant_landmarks &= {int(i) for i in allowed_landmarks}

    common = set(teacher.keys()) & set(student.keys()) & relevant_landmarks

    rot_inv = student_params["rot"].T
    render_rot_inv = render_params["rot"].T if render_params is not None else None

    def landmark_value(pose_data, idx):
        if pose_data is None:
            return None
        if idx in pose_data:
            return pose_data[idx]
        return pose_data.get(str(idx)) if isinstance(pose_data, dict) else None

    def is_finite_point(point):
        return np.all(np.isfinite(point))

    def in_frame(point):
        x, y = point
        return 0 <= x < img_width and 0 <= y < img_height

    def clamp_preview_arrow(start_px, end_px):
        vector = end_px - start_px
        length = np.linalg.norm(vector)
        if not np.isfinite(length) or length < min_arrow_px:
            return None
        if length > max_arrow_px:
            vector = vector / length * max_arrow_px
            end_px = start_px + vector
        end_px = np.array([
            np.clip(end_px[0], 0, img_width - 1),
            np.clip(end_px[1], 0, img_height - 1),
        ])
        if np.linalg.norm(end_px - start_px) < min_arrow_px:
            return None
        return tuple(np.round(end_px).astype(int))

    for j in common:
        t_val = teacher[j]
        s_val = student[j]

        dist = np.linalg.norm(t_val - s_val)

        if dist > threshold:
            scaled_t = t_val * student_params["torso_length"]
            unrotated_t = rot_inv @ scaled_t
            target_raw = unrotated_t + student_params["mid_hip"]

            scaled_s = s_val * student_params["torso_length"]
            unrotated_s = rot_inv @ scaled_s
            student_raw = unrotated_s + student_params["mid_hip"]

            start_point = None
            end_point = None
            if (
                render_pose is not None
                and render_student is not None
                and render_params is not None
                and render_rot_inv is not None
                and j in render_student
            ):
                render_landmark = landmark_value(render_pose, j)
                if render_landmark is None:
                    continue

                render_landmark = np.array(render_landmark)
                visibility = render_landmark[3] if len(render_landmark) > 3 else 1.0
                if visibility < self.visibility_threshold:
                    continue

                start_raw = render_landmark[:2]
                if not is_finite_point(start_raw):
                    continue

                start_px = np.array([start_raw[0] * img_width, start_raw[1] * img_height])
                if not in_frame(start_px):
                    continue

                scaled_preview_t = t_val * render_params["torso_length"]
                target_preview_raw = render_rot_inv @ scaled_preview_t + render_params["mid_hip"]
                end_px = np.array([
                    target_preview_raw[0] * img_width,
                    target_preview_raw[1] * img_height,
                ])

                if not is_finite_point(end_px):
                    continue

                start_point = tuple(np.round(start_px).astype(int))
                end_point = clamp_preview_arrow(start_px, end_px)
                if end_point is None:
                    continue

            corrections.append({
                "joint": j,
                "start_point": start_point,
                "end_point": end_point,
                "start_point_3d": student_raw.tolist(),
                "end_point_3d": target_raw.tolist(),
                "error_magnitude": dist
            })
            # print(corrections) #also consist 3d coords

    corrections.sort(key=lambda item: item["error_magnitude"], reverse=True)
    if max_corrections is not None:
        corrections = corrections[:max_corrections]

    return corrections
  
  def draw_arrow(self, image, corrections:list[dict]):
    """
    use cv2.arrowedLine(image, start_point, end_point, color, thickness, line_type, shift, tipLength)
    if theres correction:
    take the point coordinate of that correction (raw) for both student and teacher
    draw arrow using open cv, student -> teacher
    """
    for correction in corrections:
      start_point = correction["start_point"]
      end_point = correction["end_point"]
      if start_point is None or end_point is None:
        continue

      cv2.arrowedLine(
        img=image,
        pt1=start_point,
        pt2=end_point,
        color=(0,0,255),
        thickness=2,
        line_type=cv2.LINE_AA,
        shift=0,
        tipLength=0.2
      )
    return image
  
  def _pose_to_dict(self, pose):
    """
    Helper to convert MediaPipe pose to a dictionary of numpy arrays.
    Returns format: { id: np.array([x, y, z, visibility]) }
    """
    pose_dict = {}
    for i in range(len(pose)):
      pose_dict[i] = np.array([
        pose[i].x, 
        pose[i].y, 
        pose[i].z, 
        getattr(pose[i], "visibility", 1.0)
      ])
    return pose_dict
  
  def mark_frame(self, frame, draw_landmarks: bool = True):
    """
    Create the landmarks directly from an OpenCV frame (numpy array)
    """
    # Convert BGR (OpenCV default) to RGB
    image_3_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image from numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_3_channel)
    pose_landmarker_result = self.landmarker.detect(mp_image)
    
    # If no human is detected, return the original frame and None
    if not pose_landmarker_result.pose_landmarks:
      return frame, None, None, None
      
    pose = pose_landmarker_result.pose_landmarks[0]

    pose_dict = self._pose_to_dict(pose) 
    smoothed_pose = self.pose_filter.process(pose_dict)
    
    # Also return raw world landmarks for Unity avatar segment driving
    raw_world = None
    smoothed_world_landmarks = None
    if pose_landmarker_result.pose_world_landmarks:
      raw_world = pose_landmarker_result.pose_world_landmarks[0]
      world_dict = self._pose_to_dict(raw_world)
      smoothed_world_landmarks = self.world_pose_filter.process(world_dict)
    # ============ CLAUDE GENERATED END ============
    
    if draw_landmarks:
      annotated_image = self.draw_landmarks_on_image(image_3_channel, pose_landmarker_result)
      image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    else:
      image = frame.copy()
    return image, smoothed_pose, raw_world, smoothed_world_landmarks

#----- playback functions

  def serialize_pose(self, pose):
        """
        Converts MediaPipe landmarks or NumPy arrays into standard Python lists 
        so they can be easily saved to a JSON file.
        """
        serializable_pose = {}
        if isinstance(pose, dict):
            for k, v in pose.items():
                if isinstance(v, np.ndarray):
                    serializable_pose[int(k)] = v.tolist()
                else:
                    serializable_pose[int(k)] = list(v)
        else:
            for i, v in enumerate(pose):
                serializable_pose[i] = [v.x, v.y, getattr(v, "z", 0.0), getattr(v, "visibility", 1.0)]
        return serializable_pose
  
  def save_student_record(self, video_name, session_records):
        """
        Organizes and saves the student session data based 
        on video name and the exact time the training was taken.
        """
        import datetime
        import os

        # Create timestamp for session organization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Organize by video name, then timestamp
        base_name = os.path.basename(video_name).split('.')[0]
        save_dir = f"./student_records/{base_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = f"{save_dir}/session_{timestamp}.json"
        
        payload = {
            "video_source": video_name,
            "timestamp": timestamp,
            "records": session_records # Contains keyposes, correction angles, and arrows
        }
        
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=4)
        
        print(f"Student record saved successfully to: {file_path}")
        return file_path
  
  #----- deal with this later

  def transmit_haptic_data(self, corrections):
    """
    Takes the pose corrections, formats them into motor/vibration 
    angles, and pushes them to the ESP32.
    """
    if not corrections:
        return
        
    # Example: Extract the specific angles you need from the corrections list.
    # This logic depends entirely on how your wearable is mapped.
    pan_angle = 90  # Default neutral
    tilt_angle = 180 # Default neutral

    for c in corrections:
        # Assuming 'c' is the dictionary from your euclidean_distance or difference method
        joint_name = c.get("joint", "")
        error_val = c.get("error_magnitude", 0)
        
        # TODO: Map your specific joints to specific hardware angles here
        if joint_name == "chest_tilt":
            tilt_angle = int(180 - error_val) # example mapping
        elif joint_name == "R_shoulder":
            pan_angle = int(90 + error_val)   # example mapping

    # Format the payload exactly as your ESP32 expects it
    payload = f"{pan_angle},{tilt_angle}"

    # Safely hand it off to the background thread
    if self.ble_transmit is not None:
        self.ble_transmit.send_data(payload)

  # ============ CLAUDE GENERATED START ============
  def corrections_to_unity_format(self, corrections):
    """
    Converts euclidean_distance() correction list into the JSON payload
    that Unity's Spawner.cs / CorrectionData expects.

    Reuses the same reverse_LM mapping and quaternion conversion logic
    from video_mediapipeline.py's compare_two_images function.
    """
    # Maps mediapipe joint indices to Unity CorrectionData field names
    reverse_LM = {
        14: "r_elbow",     16: "r_forearm",
        13: "l_elbow",     15: "l_forearm",
        12: "r_shoulder",  11: "l_shoulder",
        24: "r_hip",       23: "l_hip",
        26: "r_knee",      25: "l_knee",
        28: "r_ankle",     27: "l_ankle",
    }

    # Initialize all fields to None (Unity expects all keys present)
    final: dict[str, dict[str, float] | None] = {
        "r_elbow": None,   "r_forearm": None,
        "l_elbow": None,   "l_forearm": None,
        "r_shoulder": None, "l_shoulder": None,
        "r_hip": None,     "l_hip": None,
        "r_knee": None,    "l_knee": None,
        "r_ankle": None,   "l_ankle": None,
    }

    for c in corrections:
        joint_idx = c["joint"]
        if joint_idx not in reverse_LM:
            continue

        arrow_vector = np.array(c["end_point_3d"]) - np.array(c["start_point_3d"])
        # MediaPipe coords (X-right, Y-down, Z-toward-camera)
        # → Unity coords (X-right, Y-up, Z-forward)
        arrow_vector = np.array([arrow_vector[0], -arrow_vector[1], -arrow_vector[2]])
        norm = np.linalg.norm(arrow_vector)
        if norm < 1e-6:
            continue
        arrow_vector = arrow_vector / norm

        start_vector = np.array([1.0, 0.0, 0.0])  # Unity's right direction
        rotation = R.align_vectors([arrow_vector], [start_vector])  # 2D arrays required
        q = rotation[0].as_quat()

        joint_name = reverse_LM[joint_idx]
        final[joint_name] = {
            "x": float(q[0]), "y": float(q[1]),
            "z": float(q[2]), "w": float(q[3])
        }

    return {"corrections": final}
  # ============ CLAUDE GENERATED END ============
