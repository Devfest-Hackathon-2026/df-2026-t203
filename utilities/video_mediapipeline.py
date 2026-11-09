import cv2
import numpy as np
import time
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    from mediapipe.tasks.python.vision import drawing_utils
    from mediapipe.tasks.python.vision import drawing_styles
    _HAS_TASKS_DRAWING = True
except ImportError:
    from mediapipe.python.solutions import drawing_utils
    from mediapipe.python.solutions import drawing_styles
    from mediapipe.framework.formats import landmark_pb2
    _HAS_TASKS_DRAWING = False
from scipy.spatial.transform import Rotation as R
import math
import socket

from utilities.mediapipeline import MedaiPipeline,human
from OneEuroFilter import OneEuroFilter
from pathlib import Path
import time
import asyncio



def human_analysis(pose):
    """
    Takes landmarks and generates angles for the body by making body into vectors and finding angles between them
    """
    VISIBILITY_THRESHOLD = 0.5 

    LM ={
        "nose":0,
        "l_shoulder": 11,"r_shoulder": 12,
        "l_elbow": 13,"r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16,
        "l_hip": 23, "r_hip": 24,
        "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28,
    }


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
        # v1 = v1/np.linalg.norm(v1)
        # v2 = v2/np.linalg.norm(v2)

        # for quaternions
        rotation= R.align_vectors([v1], [v2])[0]
        quart = rotation.as_quat()

        return quart

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

    #for creating human body angles
    def create_dict(pose):
        values={}
        visibility={}
        for i in range(len(pose)):
            values[i]=np.array([pose[i].x,pose[i].y,pose[i].z]) # remove z for now
            visibility[i] = getattr(pose[i],"visibility",1.0)
        return values,visibility

    def angle_if_vis(idx_list,v1,v2):
        """Return the angle of the body part if visible or not"""
        if all(visibility.get(i,1.0) >= VISIBILITY_THRESHOLD for i in idx_list):
            return find_angle(v1,v2)
        return None

    #------------
    pose_dict ,visibility= create_dict(pose)
    n = normalize_orientation(pose_dict=pose_dict) #normalize the orientation

    body_angles = {
        "R_armpit":  angle_if_vis([12, 14, 24], vec(n,12,14), vec(n,12,24)),
        "R_elbow":   angle_if_vis([12, 14, 16], vec(n,14,16), vec(n,14,12)),
        "L_armpit":  angle_if_vis([11, 13, 23], vec(n,11,13), vec(n,11,23)),
        "L_elbow":   angle_if_vis([11, 13, 15], vec(n,13,15), vec(n,13,11)),
        "chest_tilt":  angle_if_vis([11, 12], vec(n,12,11), np.array([1, 0, 0])),
        "hip_tilt":    angle_if_vis([23, 24], vec(n,24,23), np.array([1, 0, 0])),
        "R_pelvis":  angle_if_vis([24, 26], vec(n,24,23), vec(n,24,26)),
        "L_pelvis":  angle_if_vis([23, 25], vec(n,23,24), vec(n,23,25)),
        "R_knee":    angle_if_vis([24, 26, 28], vec(n,26,24), vec(n,26,28)),
        "L_knee":    angle_if_vis([23, 25, 27], vec(n,25,23), vec(n,25,27)),
    }

    return body_angles

#--- filter class
class LandmarkSmoother:
    """
    1. define the filter transformation
    2. apply the transformation to each landmark

    1 pose has 33 landmarks, 1 landmark has 3 coordinates (x,y,z), so in total 99 filters
    """
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.01):
        self.filters = {} # every coordinate has a unique filter value
        self.freq = freq
        self.min_cutoff = min_cutoff # decrease to reduce low-speed jitter
        self.beta = beta # increase to reduce high-speed lag

    def apply_to_world_landmarks(self, detection_result, timestamp_ms):
        t = timestamp_ms / 1000
        if not detection_result.pose_world_landmarks:
            return detection_result
        
        for p_idx, pose_landmarks in enumerate(detection_result.pose_world_landmarks):
            for lm_idx, landmark in enumerate(pose_landmarks):
                for coord in ['x', 'y', 'z']:
                    key = (p_idx, lm_idx, coord, 'world')
                    if key not in self.filters:
                        self.filters[key] = OneEuroFilter(freq=self.freq, mincutoff=self.min_cutoff, beta=self.beta)
                    setattr(landmark, coord, self.filters[key](getattr(landmark, coord), t))
        return detection_result

    def apply_to_landmarks(self, detection_result, timestamp_ms): # the parameters for the OneEuroFilter library
        t = timestamp_ms / 1000 # change to seconds
        if not detection_result.pose_landmarks:
            return detection_result
        
        for p_idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
            for lm_idx, landmark in enumerate(pose_landmarks):
                x_key = (p_idx, lm_idx, 'x') # tuple as the key
                if x_key not in self.filters:
                    self.filters[x_key] = OneEuroFilter(freq=self.freq, mincutoff=self.min_cutoff, beta=self.beta) # initiated a unique filter as the value
                    landmark.x = self.filters[x_key](landmark.x, t) # pass the values to the filter, just like in the docs' minimal example
                else: # if the filter is already created, just pass the values directly
                    landmark.x = self.filters[x_key](landmark.x, t)
                
                y_key = (p_idx, lm_idx, 'y') # tuple as the key
                if y_key not in self.filters:
                    self.filters[y_key] = OneEuroFilter(freq=self.freq, mincutoff=self.min_cutoff, beta=self.beta) # initiated a unique filter as the value
                    landmark.y = self.filters[y_key](landmark.y, t) # pass the values to the filter, just like in the docs' minimal example
                else: # if the filter is already created, just pass the values directly
                    landmark.y = self.filters[y_key](landmark.y, t)
                
                z_key = (p_idx, lm_idx, 'z') # tuple as the key
                if z_key not in self.filters:
                    self.filters[z_key] = OneEuroFilter(freq=self.freq, mincutoff=self.min_cutoff, beta=self.beta) # initiated a unique filter as the value
                    landmark.z = self.filters[z_key](landmark.z, t) # pass the values to the filter, just like in the docs' minimal example
                else: # if the filter is already created, just pass the values directly
                    landmark.z = self.filters[z_key](landmark.z, t)
        return detection_result
    
#---saving checkpoints to .npy
def save_checkpoints(checkpoints, video_name): 
    filename = Path(video_name).stem
    filepath = f"{filename}.npy"
    np.save(filepath, checkpoints)
    print(f"Saved {len(checkpoints)} checkpoints to {filepath}")

def load_checkpoints(filepath):
    checkpoints = np.load(filepath, allow_pickle=True) # required when loading a list of arrays
    print(f"Loaded {len(checkpoints)} checkpoints from {filepath}")
    return checkpoints

#---comparison
def normalize_3d_landmarks(pose_world_landmarks):
    """
    Converts Mediapipe 3d word landmarks into a normalized Numpy array
    1. Translates the origin to the midpoint of the hip
    
    Indices for landmarks:
    left_hip: 23, right_hip: 24 

    2. Scales the skeleton so the torso length equals 1.0
    """
    np_coords = np.array([[lm.x, lm.y, lm.z] for lm in pose_world_landmarks])

    left_hip = np_coords[23]
    right_hip = np_coords[24]
    center_hip = (left_hip+right_hip)/2.0
    translated_coords = np_coords - center_hip # move the whole body so the hip center is exact;y at (0,0,0)

    left_shoulder = np_coords[11]
    right_shoulder = np_coords[12]
    center_shoulder = (left_shoulder+right_shoulder)/2.0
    translated_coords_shoulder = center_shoulder - center_hip

    torso_length = np.linalg.norm(translated_coords_shoulder)

    if torso_length > 0:
        normalized_coords = translated_coords / torso_length
    else:
        normalized_coords = translated_coords
    
    return normalized_coords

def get_pose_difference(pose1_landmarks, pose2_landmarks):
    """
    Calculate how different are the poses
    2 sets of landmarks from 2 different poses, by using mean Euclidean distance
    """
    distances = np.linalg.norm(pose1_landmarks - pose2_landmarks, axis=1)
    return np.mean(distances)

def extract_checkpoints(video_landmarks_list, change_threshold): # this list has the landmarks of all the poses in the video
    checkpoints = []
    last_saved_pose = video_landmarks_list[0] # save the first pose as the starting checkpoint
    for current_pose in video_landmarks_list:
        diff = get_pose_difference(last_saved_pose, current_pose)
        if diff >= change_threshold:
            checkpoints.append(current_pose)
            last_saved_pose = current_pose
    return checkpoints

def compare_user_frame(user_pose, checkpoints, current_target_idx, match_threshold):
    """
    Check if user's current pose matches with target checkpoint
    Returns updated target index and boolean if the sequence is completed
    """
    if current_target_idx >= len(checkpoints):
        return current_target_idx, True
    
    target_pose = checkpoints[current_target_idx]

    diff = get_pose_difference(user_pose, target_pose)

    if diff <= match_threshold:
        current_target_idx += 1
    
    return current_target_idx, False
                    


#---basic detection
def draw_landmarks_on_image(rgb_image, detection_result):
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

def calculate_delay(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Detected FPS: {fps}")
    
    if fps == 0 or np.isnan(fps): 
        fps = 30.0 
        print("Falling back to 30 FPS.")
        
    delay = int(1000 / fps)
    return delay

#Compare with key pose

def compare_two_images(pose,picture2 = "./download.png"):
    """ TAKE TWO IMAGES AND COMPARE THE TWO, THEN SHOW"""

    pipeline=MedaiPipeline()
    image2,pose2 = pipeline.mark_image(picture2)

    t_angles, t_points = pipeline.human_analysis(pose)
    s_angles, s_points = pipeline.human_analysis(pose2)
    teacher = human(t_angles,t_points)
    student = human(s_angles,s_points)
    corrections = pipeline.difference(teacher,student,image2)
    final_corrections =[]
    for i in corrections:
        string_man = str(i["hint"]).split(":")[0]
        up = False
        if str(i).find("+") != -1: #for now check if upwards or downwards
            print("arrow is upwards")
            up = True
        

        final_corrections.append({"body":string_man,
                                  "direction": R.from_euler("x", 90, degrees=True).as_quat() if up == True else R.from_euler("x", -90, degrees=True).as_quat() })
    
    return final_corrections
    

# Example usage 2
async def check_keyposes(pose):
    """
    Check which of the images in keyposes is closest to the image: ./image.png
    """

    pipeline = MedaiPipeline()
    print("Check Keyposes")
    pose_dictionary={}
    for i,v in enumerate(pose[0]):
        pose_dictionary[i] = [v.x,v.y,v.z,v.visibility]

    with open("./keyposes/poses.json","r") as f:
        stored:dict = json.load(f)

    pose_dictionary = pipeline.normalize(pose_dictionary)

    for key, value in stored.items():
        stored[key] = pipeline.normalize(value)

    score = float("inf") #infinity
    index = None


    for i,v in stored.items():
        current_score = 0
        valid_joints = 0

        for j,values in v.items():
            vector1 = pose_dictionary[int(j)]
            vector2 = values

            if vector1[3] > pipeline.visibility_threshold and vector2[3] > pipeline.visibility_threshold:
                valid_joints +=1
                current_score+=pipeline.compare_vectors(vector1,vector2)

        if valid_joints > 7: #if image not empty
            average_score = current_score/valid_joints

            if average_score < score:
                index = i
                score = average_score


    return compare_two_images(pose[0],f"./keyposes/{index}.png")

# _, index = check_keyposes("./image.png")
# compare_two_images("./image.png",f"./keyposes/{index}.png")


"""
Logic: Every 3 seconds, check key pose
compare key pose
send coordinates of where fixes should be made
"""


#Sending unity information

def vect_to_dict(q):
    if q is None: return None
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2])
        # "w": float(q[3])
    }

def quat_to_dict(q):
    if q is None: return None
    return {
        "x": float(q[0]),
        "y": float(q[1]),
        "z": float(q[2]),
        "w": float(q[3])
    }

def human_analysis_segmentation(pose):
    """
    human analysis but with normalizing all vectors and vectors arent angles between the things but just from e.g. shoulder to elbow
    """
    VISIBILITY_THRESHOLD = 0.5
    LM ={
        "nose":0,
        "l_shoulder": 11,"r_shoulder": 12,
        "l_elbow": 13,"r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16,
        "l_hip": 23, "r_hip": 24,
        "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28,
    }

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

    #for creating human body angles
    def create_dict(pose):
        values={}
        visibility={}
        for i in range(len(pose)):
            values[i]=np.array([pose[i].x,pose[i].y,pose[i].z]) # remove z for now
            visibility[i] = getattr(pose[i],"visibility",1.0)
        return values,visibility

    def segment(n,a,b, indices):
        if all(visibility.get(i,1.0) >= VISIBILITY_THRESHOLD for i in indices):
            difference=n[b] - n[a] #difference between two points in a space
            norm =np.linalg.norm(difference)
            if norm > 0.0000001:
                return (difference/norm).tolist()
        return None
    
    def head_tilt(pose_dict):
        """ Calculate where head is tilting by measuring distance between ears and nose"""
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

        def yaw():
            YAW_MULTIPLIER = 0.8
            l_ear=  pose_dict[8][0]
            r_ear = pose_dict[7][0]
            nose = pose_dict[0][0]
            # ear_distance = (r_shoulder - l_shoulder)/2
            # middle = (l_shoulder + r_shoulder)/2
            distance_right = np.linalg.norm(nose - r_ear)
            distance_left = np.linalg.norm(nose - l_ear)
            right_bias = False
            # tilt = nose - middle
            # if (abs(r_shoulder-nose) > abs(l_shoulder - nose)):
            #     right_bias = True
            # if right_bias:
            #     angle = math.degrees(math.asin(np.clip(abs(tilt)/ear_distance,-1.0,1.0)))
            # else:
            #     angle = math.degrees(math.asin(-np.clip(abs(tilt)/ear_distance,-1.0,1.0)))
            angle = -math.degrees(math.atan2(distance_left - distance_right, distance_right + distance_left))
            quaternion = R.from_euler("y",angle*YAW_MULTIPLIER,degrees=True)
            # print("Yaw = ", angle * YAW_MULTIPLIER)
            return quaternion.as_quat()


        def pitch():
            MULTIPLIER = 1.0
            # right_side = np.array(pose_dict[7][:3]) - np.array(pose_dict[0][:3])
            # left_side  = np.array(pose_dict[8][:3]) - np.array(pose_dict[0][:3])
            # center = [1,0,0]

            # right_angle = angle_between(right_side,center) 
            # left_angle  = angle_between(left_side,center)

            # avg = (right_angle + left_angle) / 2
            # avg-=100

            nose = pose_dict[0]
            left_ear = pose_dict[7]
            right_ear = pose_dict[8]
            mid_ear_y = (left_ear[1] + right_ear[1]) / 2.0
            mid_ear_z = (left_ear[2] + right_ear[2]) / 2.0
            dy = mid_ear_y - nose[1] 
            dz = mid_ear_z - nose[2] 
            pitch_rad = math.atan2(dy, dz)
            avg = math.degrees(pitch_rad) +15

            # print("Pitch = ", avg * MULTIPLIER)
            return R.from_euler("x", avg * MULTIPLIER, degrees=True).as_quat()

        """
        Example usage, but its broken

        def roll():
            ROLL_MULTIPLIER = 1.0
            right_ear = pose_dict[7]
            left_ear  = pose_dict[8]
            right_shoulder = pose_dict[12]
            left_shoulder  = pose_dict[11]
            body_right = right_shoulder - left_shoulder  
            ear_line = right_ear - left_ear              

            # cross product gives sin of angle, dot gives cos of angle
            cross = ear_line[0] * body_right[1] - ear_line[1] * body_right[0]
            dot   = ear_line[0] * body_right[0] + ear_line[1] * body_right[1]
            final_angle = math.degrees(math.atan2(cross, dot))

            print("Roll = ", final_angle)
            return R.from_euler("z", final_angle * ROLL_MULTIPLIER, degrees=True).as_quat()
        
        """

        def roll():
            #MAKE ROLL TO DO WITH CHEST ALLIGNMENT NOT X AND Y CUZ WHAT IF THE BODY ROTATES
            ROLL_MULTIPLIER = 1.3
            right_ear = pose_dict[7]
            left_ear = pose_dict[8]

            final_angle = -math.degrees(math.atan2(right_ear[1] - left_ear[1],right_ear[0] - left_ear[0]))
            # middle_line = right_ear - left_ear
            # center_line = [1,0]
            # final_angle = angle_between(middle_line,center_line) 
            # if right_ear[1] > left_ear[1]:
            #     final_angle *= -1  
            # print("FINAL ANGLE ",final_angle)

            # print("Roll = ", final_angle * ROLL_MULTIPLIER)
            quat = R.from_euler("z",final_angle*ROLL_MULTIPLIER,degrees=True)
            return quat.as_quat()

        yaw_angle = yaw()
        pitch_angle = pitch()
        roll_angle = roll()

        return ( R.from_quat(yaw_angle)  * R.from_quat(pitch_angle) * R.from_quat(roll_angle)).as_quat().tolist()
        # return ( R.from_quat(yaw_angle)).as_quat().tolist()
        # return ( R.from_quat(pitch_angle)).as_quat().tolist()
        # return (R.from_quat(roll_angle)).as_quat().tolist()


    
    pose_dict, visibility =create_dict(pose)
    # n=normalize_orientation(pose_dict) #need to stop using this to allow for rotations

    n=pose_dict

    mid_hip = (n[23] + n[24]) / 2
    mid_shoulder = (n[11] + n[12]) / 2

    return {
        "R_upper_arm":  segment(n, 12, 14, [12, 14]),
        "L_upper_arm":  segment(n, 11, 13, [11, 13]),
        "R_forearm":    segment(n, 14, 16, [14, 16]),
        "L_forearm":    segment(n, 13, 15, [13, 15]),
        "R_thigh":  segment(n, 24, 26, [24, 26]),
        "L_thigh":  segment(n, 23, 25, [23, 25]),
        "R_shin":  segment(n, 26, 28, [26, 28]),
        "L_shin":  segment(n, 25, 27, [25, 27]),
        "spine":        (mid_shoulder - mid_hip).tolist() if np.linalg.norm(mid_shoulder - mid_hip) > 1e-6 else None,
        "head_tilt": head_tilt(pose_dict)
    }


async def main():
    """
    Processing loop:
    1. Capture each frame of the video by using cv2
    2. Convert BGR (original and cv2 compatibility) -> RGB (mediapipe compatibility)
    3. Detect pose landmark for each frame
    4. Convert back to BGR for cv2 display
    5. Loop for all frames
    
    TODO: 2 parts: Extract checkpoints from video and side-by-size view user comparison
    """
    # load the task (model)
    model_path = "./pose_landmarker_full.task"
    BaseOptions = python.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode = VisionRunningMode.VIDEO
        )
    
    cap = cv2.VideoCapture(0)
    delay = calculate_delay(cap)

    start_time = time.time()
    last_keypose_check = 0.0
    updatetime = 2


    smoother = LandmarkSmoother(min_cutoff=0.05, beta=0.80)

    #socket sending for unity
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ("127.0.0.1",5052)

    #second socket for corrections
    correction_server_address = ("127.0.0.1",5053)

    with PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read() # capture frame by frame, ret is return, will return True if frame is read
            if not ret:
                print("exiting..") # ret will turn false when the video finished 
                break

            if len(frame.shape) == 2:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # if the image is 2D (grayscale), convert GRAY to RGB
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # if the image is 3D (standard color), convert BGR to RGB
            
            mediapipe_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # convert np array to mediapipe Image object
            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)
            timestamp_ms += 1

            try: 
                detection_result = detector.detect_for_video(mediapipe_frame, timestamp_ms) # tracks landmarks across time
                filtered_result = smoother.apply_to_landmarks(detection_result, timestamp_ms)

                filtered_world = smoother.apply_to_world_landmarks(detection_result,timestamp_ms) #filter for world landmarks and stuff

                if filtered_result.pose_landmarks:
                    #For IK analysis
                    # data = []
                    # for lm in filtered_result.pose_landmarks[0]:
                    #     data.append({"x":lm.x,"y": lm.y,"z": lm.z})
                    
                    data = human_analysis_segmentation(filtered_world.pose_world_landmarks[0])
                    
                    if current_time - last_keypose_check >= updatetime:
                        last_keypose_check = current_time
                        corrections = await check_keyposes(filtered_result.pose_landmarks)
                        if len(corrections)>0:
                            correction_payload = {
                                "corrections": {corrections[0]["body"]:quat_to_dict(corrections[0]["direction"].tolist())}
                            }
                        
                            sock.sendto(json.dumps(correction_payload).encode(), correction_server_address)



                    payload = {
                        "segments" : {k:quat_to_dict(v) if k == "head_tilt" else vect_to_dict(v) 
                                      for k,v in data.items() if v is not None}
                    }

                    sock.sendto(json.dumps(payload).encode(), server_address)
                    # print(data)


                annotated_frame = draw_landmarks_on_image(mediapipe_frame.numpy_view(), filtered_result)
                bgr_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
                cv2.imshow('Pose Estimation', bgr_annotated_frame)
            except Exception as e:
                print("Error as ", e)
            
            if cv2.waitKey(1) == ord('q'): # waitKey() receives int(delay)
                break

        cap.release()
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    asyncio.run(main())

"""
TODO: 
Add turning left or right?
Add constrictions to things like elbows and stuff to make it more accurate
"""