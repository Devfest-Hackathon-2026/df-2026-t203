"""
Anything to do with preprocessing
"""
import os
import json
import cv2
import numpy as np
from utilities.video_mediapipeline import get_pose_difference, normalize_3d_landmarks, human_analysis
from utilities.mediapipeline import MedaiPipeline
from utilities.keyposes import JOINT_WEIGHTS, keypose_indices, normalize_energy
from scipy.ndimage import gaussian_filter1d

def generate_keyposes(video_name:str, output_path:str = "keyposes/teacher_keyposes.json", output_path_time:str = "keyposes/teacher_timestamps.json") -> None:
    """
    Generates keyposes from a videopath and saves them in keyposes/teacher_keyposes
    
    :param video_name: Video path 
    :type video_name: str

    :param output_path: Path to output the keyposes json file 
    :type output_path: str
    """
    pipeline = MedaiPipeline()
    cap = cv2.VideoCapture(video_name)
    keyposes = {}
    timestamps = {}
    pose_id = 0
    last_saved_pose_landmarks = None

    DISTANCE = 0.30 # distance moved for new keypose to be made
    COOLDOWN = 30
    cooldown_timer = 0
    print("process video")
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = time_ms / 1000.0

        if cooldown_timer > 0:
            cooldown_timer -=1
            continue

        #detect pose landmarks
        _, pose, world_pose, smoothed_world = pipeline.mark_frame(frame)

        if world_pose is not None and smoothed_world is not None:
            #normalize
            current_normalized = normalize_3d_landmarks(world_pose)

            #initial pose
            if last_saved_pose_landmarks is None:
                keyposes[str(pose_id)] = pipeline.serialize_pose(smoothed_world)
                timestamps[str(pose_id)] = timestamp_sec
                last_saved_pose_landmarks = current_normalized
                cv2.imwrite(f"keyposes/pose_{pose_id}.png", frame)
                print(f"Logged initial pose {pose_id}")
                pose_id+=1
                cooldown_timer = COOLDOWN
            
            #new poses
            else:
                #Calculate differece
                diff = get_pose_difference(last_saved_pose_landmarks, current_normalized)

                if diff>= DISTANCE:
                    keyposes[str(pose_id)] = pipeline.serialize_pose(smoothed_world)
                    last_saved_pose_landmarks = current_normalized
                    timestamps[str(pose_id)] = timestamp_sec
                    cv2.imwrite(f"keyposes/pose_{pose_id}.png", frame)
                    print("Logged pose id ",pose_id)
                    pose_id+=1
                    cooldown_timer = COOLDOWN

    cap.release()


    #save the content into files
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    os.makedirs(os.path.dirname(output_path_time),exist_ok=True)


    with open(output_path,"w") as f:
        json.dump(keyposes,f,indent=4)
    with open(output_path_time,"w") as f:
        json.dump(timestamps,f,indent=4)
    print("Keyposes created, number of keyposes: ",len(keyposes))

#Loop through video and per frame have all values for 33 points

def generate_keyposes_new(video_name:str, output_path:str = "keyposes/teacher_keyposes.json", output_path_time:str = "keyposes/teacher_timestamps.json", frame_skip:int = 2):
    """
    Energy-based keypose extraction for martial arts, dance, and dynamic movement videos.

    For every frame:
      1. Compute joint translational velocity (vJ) and angular velocity (vT)
      2. Apply Gaussian smoothing
      3. Normalize each signal to [0, 1]
      4. Calculate combined energy: E = wT*vT + wJ*vJ (weighted by joint importance)
      5. Find movement peaks (moments of maximum motion), then locate the
         nearest valley after each peak — this is the "held pose" right after
         a significant movement, which is the actual keypose.

    :param video_name: Path for Video
    :type video_name: str
    :param frame_skip: Process every Nth frame for speed (default 2). Set to 1 to process all frames.
    :type frame_skip: int
    """
    pipeline = MedaiPipeline(enable_ble=False, ema_alpha=0.4)  # No BLE needed for preprocessing
    cap = cv2.VideoCapture(video_name)

    joints:dict = {}  # structure is {frame_id: {0:.., -> 33:...}}
    frame_id = 0
    print("Processing video")
    frame_times = {}

    raw_joints_list = []
    raw_angles_list = []
    valid_frame_ids = []
    last_valid_angles = {}

    process_counter = 0

    while cap.isOpened():
        frame_id += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping
        process_counter += 1
        if process_counter % frame_skip != 0:
            continue

        time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = time_ms / 1000.0  
        frame_times[frame_id] = timestamp_sec

        # Downscale for speed — 
        frame = cv2.resize(frame, (640, 480))

        # Detect pose landmarks
        _, pose, world_pose, smoothed_world = pipeline.mark_frame(frame)

        if world_pose is not None and smoothed_world is not None:
            joints[str(frame_id)] = pipeline.serialize_pose(world_pose)
            frame_joints = [smoothed_world[key][:3] for key in sorted(smoothed_world.keys())]
            raw_joints_list.append(frame_joints)
            valid_frame_ids.append(frame_id)

            # Compute joint angles once per frame
            angles_dict = human_analysis(world_pose)
            frame_angles = []
            for key in sorted(angles_dict.keys()):
                val = angles_dict[key]
                if val is None:
                    # use the last known valid angle (or identity quaternion if first frame)
                    val = last_valid_angles.get(key, [0.0, 0.0, 0.0, 1.0])
                else:
                    last_valid_angles[key] = val
                frame_angles.append(val)

            raw_angles_list.append(frame_angles)

    cap.release()
    print(f"Processed {len(valid_frame_ids)} frames out of {frame_id} total (skip={frame_skip})")

    if len(raw_joints_list) < 3:
        print("Not enough valid frames to detect keyposes")
        return

    joints_array = np.array(raw_joints_list)  # shape: (N, 33, 3)
    angles_array = np.array(raw_angles_list)   # shape: (N, num_angles, 4)

    # velocities ---

    # velocity change
    xyz_differences = np.diff(joints_array, axis=0)            
    joint_distances = np.linalg.norm(xyz_differences, axis=2)   # (N-1, 33)
    # Apply per-joint weights so that shoulders/hips/knees matter more than face
    weighted_distances = joint_distances * JOINT_WEIGHTS[np.newaxis, :]
    vJ = weighted_distances.sum(axis=1) / JOINT_WEIGHTS.sum()   # weighted mean

    # Angular velocity
    angle_differences = np.diff(angles_array, axis=0)
    quaternion_distances = np.linalg.norm(angle_differences, axis=2)
    vT = quaternion_distances.mean(axis=1)

    # Gaussian smoothing 
    sigma = 3  # slightly wider kernel for smoother signal on dynamic movements
    filtered_vJ = gaussian_filter1d(vJ, sigma=sigma)
    filtered_vT = gaussian_filter1d(vT, sigma=sigma)

    print("Gaussian smoothing done")

    # Combine angular and translational motion before finding held poses.
    energy = 0.4 * normalize_energy(filtered_vT) + 0.6 * normalize_energy(filtered_vJ)
    cap_info = cv2.VideoCapture(video_name)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()
    effective_fps = (fps if fps > 0 else 30.0) / frame_skip
    keyframe_actual_indices = keypose_indices(energy, effective_fps)

    timestamps = {}
    keyposes = {}
    for idx, pose_frame_idx in enumerate(keyframe_actual_indices):
        actual_frame_id = valid_frame_ids[pose_frame_idx]
        timestamps[idx] = frame_times[actual_frame_id]
        keyposes[idx] = joints[str(actual_frame_id)]

    # Save teacher trajectories between keyposes for DTW comparison.
    # Each trajectory is the sequence of poses from keypose N to keypose N+1.
    teacher_trajectories = {}
    for i in range(len(keyframe_actual_indices) - 1):
        start_idx = keyframe_actual_indices[i]
        end_idx = keyframe_actual_indices[i + 1]
        trajectory = []
        for j in range(start_idx, end_idx + 1):
            fid = valid_frame_ids[j]
            if str(fid) in joints:
                trajectory.append(joints[str(fid)])
        teacher_trajectories[str(i)] = trajectory

    out_dir = os.path.dirname(output_path) or "keyposes"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path_time) or "keyposes", exist_ok=True)

    trajectory_path = os.path.join(out_dir, "teacher_trajectories.json")
    with open(trajectory_path, "w") as f:
        json.dump(teacher_trajectories, f)
    print(f"Teacher trajectories saved ({len(teacher_trajectories)} segments)")
    # AI GENERATED END — teacher trajectory saving

    print("Done calculating energy")

    with open(output_path, "w") as f:
        json.dump(keyposes, f, indent=4)
    with open(output_path_time, "w") as f:
        json.dump(timestamps, f, indent=4)
    print("Keyposes created, number of keyposes: ", len(keyposes))

    print("Saving keyframe images...")
    cap2 = cv2.VideoCapture(video_name)
    for idx, pose_frame_idx in enumerate(keyframe_actual_indices):
        actual_frame_id = valid_frame_ids[pose_frame_idx]
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(actual_frame_id) - 1)
        ret, frame = cap2.read()
        if ret:
            cv2.imwrite(os.path.join(out_dir, f"pose_{idx}.png"), frame)
    cap2.release()
    print("All keyframe images saved!")
