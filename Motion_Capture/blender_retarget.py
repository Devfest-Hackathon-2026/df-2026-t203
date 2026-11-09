"""Stage 2 of the single-video pipeline: landmark JSON + rigged FBX -> baked Blender animation.

This runs *inside* Blender (it imports ``bpy``), driven by ``run_pipeline.py`` or
by hand:

    blender --background --python blender_retarget.py -- \
        --landmarks output/Water/landmarks.json \
        --model    ../avatar-models/avatar_model.fbx \
        --blend-out output/Water/Water.blend

How the retarget works
----------------------
The rig is a Mixamo skeleton, so bones are addressed by their bare name
(``LeftHandIndex1``) and the namespace prefix on the file (``mixamorig4:``) is
detected at import time.

Every driven bone is solved as a **world rotation**, never as a position. The
avatar is not the person in the video and has different limb proportions, so
matching joint *positions* would put the hands in the wrong place; matching joint
*directions* transfers the pose and lets the avatar keep its own body. Two kinds
of solve produce that rotation:

* **Direction bones** - clavicles, upper arms, forearms and every finger bone.
  The measured direction is one landmark minus another (elbow minus shoulder,
  fingertip minus knuckle) and the bone is swung from its rest direction onto it
  by the shortest arc. Swing leaves roll undefined, so the parent's own rotation
  is applied first and the swing is taken from there - that is what keeps a
  finger rolling with the palm instead of hanging onto its T-pose roll.

* **Basis bones** - hips, upper chest, head and both hands. An orthonormal frame
  is built from the landmarks (for a hand: wrist -> middle knuckle, and pinky
  knuckle -> index knuckle), the same frame is built from the rig's rest pose,
  and the bone's world rotation becomes ``B_frame * B_rest^-1 * R_rest``. This
  solves roll as well as direction, which is why the hands are done this way:
  palm orientation carries meaning in JSL.

Each solved rotation is keyframed onto an empty; the bone gets a world-space
Copy Rotation constraint pointed at that empty; everything is then baked to plain
keyframes so the animation survives export and the helpers can be thrown away.

What this does not do
---------------------
* No root motion. MediaPipe world landmarks are hip-centred, so a single
  uncalibrated camera gives no usable global translation; the avatar stays put.
* No legs. Nothing in a signing video constrains them usefully, so they keep the
  rest pose.
* No face, mouth shapes or eye gaze - and mouth shapes are part of JSL, so the
  output of this pipeline is not a complete sign on its own.
* No spine curvature. MediaPipe has no landmark between hips and shoulders, so
  the torso bends at two joints (hips and upper chest) and nowhere else.
* Nothing here is reviewed JSL. Anything this produces is unverified motion
  copied off a video, and must be labelled as such.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# --- MediaPipe pose landmark indices (mp.solutions.pose.PoseLandmark) --------
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Pose landmark index pairs that swap identity when the footage is mirrored.
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [
    (index, index + 1) for index in range(11, 33, 2)
]

# --- MediaPipe hand landmark indices ----------------------------------------
HAND_WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# Each finger bone spans two hand landmarks: (its head, its tail).
FINGER_SEGMENTS = {
    "HandThumb1": (1, 2),
    "HandThumb2": (2, 3),
    "HandThumb3": (3, 4),
    "HandIndex1": (5, 6),
    "HandIndex2": (6, 7),
    "HandIndex3": (7, 8),
    "HandMiddle1": (9, 10),
    "HandMiddle2": (10, 11),
    "HandMiddle3": (11, 12),
    "HandRing1": (13, 14),
    "HandRing2": (14, 15),
    "HandRing3": (15, 16),
    "HandPinky1": (17, 18),
    "HandPinky2": (18, 19),
    "HandPinky3": (19, 20),
}

REQUIRED_BONES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "RightUpLeg",
]

# The palm's rest frame is measured from these three knuckle bones, so the hand
# solve cannot run without them. They are only required when hands are solved:
# a Mixamo "No Fingers" rig lacks them but is still usable with --no-hands.
PALM_REFERENCE_SUFFIXES = ("HandMiddle1", "HandIndex1", "HandPinky1")

IDENTITY = Matrix.Identity(3)


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------


def mp_to_blender(point, depth_scale: float = 1.0) -> Vector:
    """Remap one MediaPipe world landmark into Blender's axes.

    MediaPipe: +x image-right, +y image-down, +z away from the camera.
    Blender here: +x is the subject's left, +z up, +y away from the camera -
    i.e. a subject facing the camera faces -Y, which is how the Mixamo model
    imports (its index knuckles sit at -Y, its left shoulder at +X).
    """
    return Vector((point[0], point[2] * depth_scale, -point[1]))


def orthonormal_basis(primary: Vector, secondary: Vector) -> Matrix:
    """Right-handed 3x3 basis whose Y column is `primary`, X column along `secondary`.

    `secondary` only has to be roughly perpendicular to `primary`; its component
    along `primary` is removed. The columns are the basis axes, so `basis @ v`
    takes basis-local coordinates out to world.
    """
    y_axis = primary.normalized()
    x_axis = (secondary - y_axis * secondary.dot(y_axis)).normalized()
    z_axis = x_axis.cross(y_axis)
    return Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x),
            (x_axis.y, y_axis.y, z_axis.y),
            (x_axis.z, y_axis.z, z_axis.z),
        )
    )


def basis_delta(frame_basis: Matrix, rest_basis: Matrix) -> Matrix:
    """World rotation that carries the rest frame onto the measured one."""
    return frame_basis @ rest_basis.transposed()


def signed_angle(from_vector: Vector, to_vector: Vector, axis: Vector) -> float:
    """Angle from one vector to another, signed about `axis` (radians)."""
    normal = axis.normalized()
    return math.atan2(from_vector.cross(to_vector).dot(normal), from_vector.dot(to_vector))


def swing_delta(rest_direction: Vector, measured_direction: Vector, parent_delta: Matrix = IDENTITY) -> Matrix:
    """World rotation that points a bone along `measured_direction`.

    The parent's own rotation is applied to the rest direction first, then the
    shortest arc from there to the measurement is added. Doing it in that order
    means the bone inherits its parent's roll (a finger rolls with the palm)
    instead of keeping the roll it had in the T-pose.
    """
    rotated_rest = parent_delta @ rest_direction
    if rotated_rest.length < 1e-8 or measured_direction.length < 1e-8:
        return parent_delta
    swing = rotated_rest.rotation_difference(measured_direction).to_matrix()
    return swing @ parent_delta


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget MediaPipe landmarks from one video onto a Mixamo-rigged avatar.",
    )
    parser.add_argument("--landmarks", required=True, type=Path, help="landmarks.json from extract_landmarks.py.")
    parser.add_argument("--model", required=True, type=Path, help="Rigged FBX to animate (avatar_model.fbx).")
    parser.add_argument("--blend-out", required=True, type=Path, help="Where to save the .blend file.")
    parser.add_argument("--export-fbx", type=Path, default=None, help="Also export the animated rig to this FBX path.")
    parser.add_argument("--export-glb", type=Path, default=None, help="Also export to this .glb path (for the web player).")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the capture left/right. Use when the footage is flipped relative to the signer.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help=(
            "Scale MediaPipe's depth axis for the body before solving. Monocular depth is the "
            "least reliable axis; below 1.0 flattens it towards the camera plane."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=1, help="Scene frame the first video frame maps to.")
    parser.add_argument("--no-hands", action="store_true", help="Skip the hands and fingers (body only).")
    parser.add_argument("--no-head", action="store_true", help="Skip the head and neck solve.")
    parser.add_argument(
        "--no-roll-calibration",
        action="store_true",
        help=(
            "Keep MediaPipe's head roll as measured. By default the clip's median roll is "
            "subtracted, because MediaPipe's world landmarks carry a constant roll bias."
        ),
    )
    parser.add_argument(
        "--keep-helpers",
        action="store_true",
        help="Keep the target empties in the scene after baking, to inspect the solve.",
    )
    parser.add_argument(
        "--render-preview",
        type=Path,
        default=None,
        help="Render a few frames as PNGs into this folder, as a quick visual check.",
    )
    parser.add_argument("--preview-frames", type=int, default=4, help="How many frames --render-preview renders.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene / model setup
# ---------------------------------------------------------------------------


def import_model(model_path: Path):
    """Import the rigged FBX and return its armature object."""
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if hasattr(bpy.ops.wm, "fbx_import"):
        bpy.ops.wm.fbx_import(filepath=str(model_path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(model_path))

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise SystemExit(f"No armature found in {model_path.name} - is this file actually rigged?")
    if len(armatures) > 1:
        print(f"[retarget] {len(armatures)} armatures imported, using '{armatures[0].name}'")
    return armatures[0]


def build_bone_lookup(armature, need_hands: bool = True) -> dict:
    """Map a bare Mixamo bone name to the real bone name, namespace prefix and all."""
    lookup: dict = {}
    for bone in armature.data.bones:
        lookup.setdefault(bone.name.split(":")[-1], bone.name)
    required = list(REQUIRED_BONES)
    if need_hands:
        required += [f"{side}{suffix}" for side in ("Left", "Right") for suffix in PALM_REFERENCE_SUFFIXES]
    missing = [name for name in required if name not in lookup]
    if missing:
        hint = " A rig without finger bones can still be used with --no-hands." if need_hands else ""
        raise SystemExit(
            "The rig is missing bones this retarget needs: "
            + ", ".join(missing)
            + ". It expects a Mixamo-style skeleton."
            + hint
        )
    return lookup


def setup_scene(fps: float, frame_start: int, frame_end: int) -> None:
    scene = bpy.context.scene
    rounded_fps = max(1, int(round(fps)))
    scene.render.fps = rounded_fps
    scene.render.fps_base = rounded_fps / fps if fps > 0 else 1.0
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)


# ---------------------------------------------------------------------------
# Landmark data
# ---------------------------------------------------------------------------


def load_landmarks(path: Path, mirror: bool) -> dict:
    if not path.is_file():
        raise SystemExit(f"Landmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("frames"):
        raise SystemExit(f"{path} contains no frames.")
    if not any(frame.get("pose_world") for frame in data["frames"]):
        raise SystemExit(f"{path} has no pose landmarks in any frame - nothing to retarget.")
    if mirror:
        mirror_landmarks(data)
    return data


def mirror_landmarks(data: dict) -> None:
    """Reflect the capture across the image's vertical axis and relabel the sides.

    Reflecting the points alone would turn the signer into their own mirror
    image; swapping the left/right identities too turns it back into a real
    person seen the other way round, which is what a flipped recording is.
    """
    for frame in data["frames"]:
        pose = frame.get("pose_world")
        if pose is not None:
            for point in pose:
                point[0] = -point[0]
            for left_index, right_index in POSE_MIRROR_PAIRS:
                pose[left_index], pose[right_index] = pose[right_index], pose[left_index]
        hands = frame.get("hands") or {}
        for points in hands.values():
            for point in points:
                point[0] = -point[0]
        frame["hands"] = {("Right" if label == "Left" else "Left"): points for label, points in hands.items()}


# ---------------------------------------------------------------------------
# Rest pose measurements
# ---------------------------------------------------------------------------


def rest_head(armature, bone_name: str) -> Vector:
    """World-space head position of a bone in the rest pose."""
    return armature.matrix_world @ armature.data.bones[bone_name].head_local


def rest_tail(armature, bone_name: str) -> Vector:
    return armature.matrix_world @ armature.data.bones[bone_name].tail_local


def rest_direction(armature, bone_name: str) -> Vector:
    return rest_tail(armature, bone_name) - rest_head(armature, bone_name)


def rest_rotation(armature, bone_name: str) -> Matrix:
    """World-space rotation of a bone in the rest pose, with scale removed."""
    matrix = armature.matrix_world @ armature.data.bones[bone_name].matrix_local
    return matrix.to_quaternion().to_matrix()


def measure_rest(armature, bones: dict, need_hands: bool = True) -> dict:
    """Everything about the rig's rest pose that the solve compares against.

    The rig's reference points are chosen to mean the same thing as the
    MediaPipe landmarks they are matched with: shoulder centre is the midpoint of
    the two upper-arm joints (landmarks 11/12), hip centre the midpoint of the
    two thigh joints (23/24), and the head's up axis is the Head bone itself -
    deliberately not Head -> HeadTop_End, which on a Mixamo rig leans about 19
    degrees forward and would tip the avatar's head back by that much.
    """
    shoulder_center = (rest_head(armature, bones["LeftArm"]) + rest_head(armature, bones["RightArm"])) / 2.0
    hip_center = (rest_head(armature, bones["LeftUpLeg"]) + rest_head(armature, bones["RightUpLeg"])) / 2.0
    torso_up = shoulder_center - hip_center
    shoulder_axis = rest_head(armature, bones["LeftArm"]) - rest_head(armature, bones["RightArm"])
    hip_axis = rest_head(armature, bones["LeftUpLeg"]) - rest_head(armature, bones["RightUpLeg"])

    bases = {
        "hips": orthonormal_basis(torso_up, hip_axis),
        "chest": orthonormal_basis(torso_up, shoulder_axis),
        "head": orthonormal_basis(rest_direction(armature, bones["Head"]), shoulder_axis),
    }
    # The palm frame needs the knuckle bones, which a finger-less rig does not
    # have; build_bone_lookup has already guaranteed them whenever hands are on.
    for side in ("Left", "Right") if need_hands else ():
        bases[f"{side}Hand"] = orthonormal_basis(
            rest_head(armature, bones[f"{side}HandMiddle1"]) - rest_head(armature, bones[f"{side}Hand"]),
            rest_head(armature, bones[f"{side}HandIndex1"]) - rest_head(armature, bones[f"{side}HandPinky1"]),
        )

    directions = {}
    for side in ("Left", "Right"):
        for bare_name in (f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm"):
            directions[bare_name] = rest_direction(armature, bones[bare_name])
        for suffix in FINGER_SEGMENTS:
            bare_name = f"{side}{suffix}"
            if bare_name in bones:
                directions[bare_name] = rest_direction(armature, bones[bare_name])

    rotations = {
        bare_name: rest_rotation(armature, bone_name)
        for bare_name, bone_name in bones.items()
    }
    return {"bases": bases, "directions": directions, "rotations": rotations}


# ---------------------------------------------------------------------------
# Helper objects and constraints
# ---------------------------------------------------------------------------


def new_orientation_empty(name: str):
    empty = bpy.data.objects.new(f"solve_{name}", None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.05
    empty.rotation_mode = "QUATERNION"
    bpy.context.scene.collection.objects.link(empty)
    return empty


def keyframe_rotation(empty, rotation: Matrix, frame: int, previous: dict) -> None:
    """Keyframe a world rotation, keeping the quaternion on the same hemisphere.

    Without the sign fix a quaternion that flips sign between two frames makes
    Blender interpolate the long way round, and the bone spins a full turn.
    """
    quaternion = rotation.to_quaternion()
    last = previous.get(empty.name)
    if last is not None and quaternion.dot(last) < 0.0:
        quaternion.negate()
    previous[empty.name] = quaternion
    empty.rotation_quaternion = quaternion
    empty.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def add_copy_rotation(pose_bone, target, influence: float = 1.0) -> None:
    for constraint in list(pose_bone.constraints):
        pose_bone.constraints.remove(constraint)
    constraint = pose_bone.constraints.new("COPY_ROTATION")
    constraint.target = target
    constraint.target_space = "WORLD"
    constraint.owner_space = "WORLD"
    constraint.influence = influence


def select_bone(pose_bone, selected: bool) -> None:
    """Select a bone, on either side of the Blender 4.x -> 5.x API change.

    Blender 5 moved selection onto PoseBone; before that it lived on the Bone.
    """
    if hasattr(pose_bone, "select"):
        pose_bone.select = selected
    else:
        pose_bone.bone.select = selected


def bake_bones(armature, bone_names, frame_start: int, frame_end: int) -> None:
    """Bake the constrained motion of `bone_names` to keyframes and drop the constraints."""
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        select_bone(pose_bone, pose_bone.name in bone_names)
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def delete_objects(objects) -> None:
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def head_frame(pose, depth_scale: float) -> tuple:
    """The measured face-forward axis and ear axis for one frame."""
    ear_center = (mp_to_blender(pose[LEFT_EAR], depth_scale) + mp_to_blender(pose[RIGHT_EAR], depth_scale)) / 2.0
    forward = mp_to_blender(pose[NOSE], depth_scale) - ear_center
    side = mp_to_blender(pose[LEFT_EAR], depth_scale) - mp_to_blender(pose[RIGHT_EAR], depth_scale)
    return forward, side


def head_roll_bias(data: dict, depth_scale: float) -> float:
    """The clip's median head roll, which is mostly a MediaPipe artefact (radians).

    MediaPipe's *world* landmarks carry a roll bias about the camera axis that its
    image-space landmarks do not: on the sample clip the ear axis reads about 16
    degrees off level in 3D while the same ears are within a couple of degrees of
    level in the image. Left alone, that becomes an avatar whose head is
    permanently cocked to one side.

    Taking the median over the clip and subtracting it keeps whatever head tilt
    the signer actually does - only the constant part is removed. The assumption
    is that the signer's *average* head over the clip is upright, which is why
    --no-roll-calibration exists for footage where it is not.
    """
    rolls = []
    for frame in data["frames"]:
        pose = frame.get("pose_world")
        if pose is None:
            continue
        forward, side = head_frame(pose, depth_scale)
        if forward.length < 1e-6 or side.length < 1e-6:
            continue
        forward = forward.normalized()
        reference_up = Vector((0.0, 0.0, 1.0)) - forward * forward.z
        if reference_up.length < 1e-6:  # looking straight up or down; roll is undefined
            continue
        rolls.append(signed_angle(reference_up.normalized(), forward.cross(side).normalized(), forward))
    return statistics.median(rolls) if rolls else 0.0


def solve_frame_body(pose, rest: dict, depth_scale: float, no_head: bool, roll_bias: float = 0.0) -> dict:
    """World rotation deltas for the body bones on one frame.

    Returns a dict of bare bone name -> world rotation delta, plus the entries
    "chest" and each hand's parent delta that later stages build on.
    """
    point = lambda index: mp_to_blender(pose[index], depth_scale)
    left_shoulder, right_shoulder = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    left_hip, right_hip = point(LEFT_HIP), point(RIGHT_HIP)
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    torso_up = shoulder_center - hip_center

    deltas = {
        "Hips": basis_delta(orthonormal_basis(torso_up, left_hip - right_hip), rest["bases"]["hips"]),
        "chest": basis_delta(
            orthonormal_basis(torso_up, left_shoulder - right_shoulder), rest["bases"]["chest"]
        ),
    }
    deltas["Spine1"] = deltas["chest"]
    deltas["Spine2"] = deltas["chest"]

    if not no_head:
        face_forward, face_side = head_frame(pose, depth_scale)
        head_basis = orthonormal_basis(face_forward.cross(face_side), face_side)
        if roll_bias:
            head_basis = Matrix.Rotation(-roll_bias, 3, face_forward) @ head_basis
        deltas["Head"] = basis_delta(head_basis, rest["bases"]["head"])
        deltas["Neck"] = deltas["Head"]

    for side, shoulder, elbow, wrist in (
        ("Left", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        ("Right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ):
        clavicle = swing_delta(
            rest["directions"][f"{side}Shoulder"], point(shoulder) - shoulder_center, deltas["chest"]
        )
        upper_arm = swing_delta(rest["directions"][f"{side}Arm"], point(elbow) - point(shoulder), clavicle)
        forearm = swing_delta(rest["directions"][f"{side}ForeArm"], point(wrist) - point(elbow), upper_arm)
        deltas[f"{side}Shoulder"] = clavicle
        deltas[f"{side}Arm"] = upper_arm
        deltas[f"{side}ForeArm"] = forearm
    return deltas


def solve_frame_hand(points, side: str, rest: dict, bones: dict) -> dict:
    """World rotation deltas for one hand and its fingers on one frame.

    MediaPipe's hand world landmarks are wrist-relative but use the same axes as
    the pose landmarks, so a difference between two of them is already a world
    direction - no anchoring or scaling is needed to solve rotations from them.
    """
    hand = [mp_to_blender(point) for point in points]
    palm = basis_delta(
        orthonormal_basis(hand[MIDDLE_MCP] - hand[HAND_WRIST], hand[INDEX_MCP] - hand[PINKY_MCP]),
        rest["bases"][f"{side}Hand"],
    )
    deltas = {f"{side}Hand": palm}
    for suffix, (head_index, tail_index) in FINGER_SEGMENTS.items():
        bare_name = f"{side}{suffix}"
        if bare_name not in bones:
            continue
        deltas[bare_name] = swing_delta(
            rest["directions"][bare_name], hand[tail_index] - hand[head_index], palm
        )
    return deltas


def solve(armature, bones: dict, data: dict, args: argparse.Namespace) -> tuple:
    """Keyframe a target empty per driven bone, then constrain the bones to them."""
    rest = measure_rest(armature, bones, need_hands=not args.no_hands)
    targets: dict = {}
    previous_quaternions: dict = {}
    hand_frames = {"Left": 0, "Right": 0}

    roll_bias = 0.0
    if not args.no_head and not args.no_roll_calibration:
        roll_bias = head_roll_bias(data, args.depth_scale)
        print(f"[retarget] removing {math.degrees(roll_bias):.1f} degrees of constant head roll")

    for offset, frame in enumerate(data["frames"]):
        scene_frame = args.start_frame + offset
        deltas: dict = {}

        pose = frame.get("pose_world")
        if pose is not None:
            deltas.update(solve_frame_body(pose, rest, args.depth_scale, args.no_head, roll_bias))

        if not args.no_hands:
            for side in ("Left", "Right"):
                points = (frame.get("hands") or {}).get(side)
                if points is None:
                    continue
                hand_frames[side] += 1
                deltas.update(solve_frame_hand(points, side, rest, bones))

        for bare_name, delta in deltas.items():
            if bare_name == "chest":  # an intermediate, not a bone
                continue
            if bare_name not in targets:
                targets[bare_name] = new_orientation_empty(bare_name)
            keyframe_rotation(
                targets[bare_name], delta @ rest["rotations"][bare_name], scene_frame, previous_quaternions
            )

    # The neck follows the head only partly, and the lower chest follows the
    # upper chest only partly, so a turn is shared across the joints instead of
    # snapping at one of them. The head and upper chest themselves are solved in
    # world space, so their final orientation is unaffected by this.
    partial_influence = {"Neck": 0.4, "Spine1": 0.5}
    pose_bones = armature.pose.bones
    driven_bones = set()
    for bare_name, target in targets.items():
        bone_name = bones[bare_name]
        add_copy_rotation(pose_bones[bone_name], target, influence=partial_influence.get(bare_name, 1.0))
        driven_bones.add(bone_name)

    total = len(data["frames"])
    print(f"[retarget] solved {len(driven_bones)} bones over {total} frames")
    if not args.no_hands:
        for side in ("Left", "Right"):
            print(f"[retarget] {side.lower()} hand present on {hand_frames[side]}/{total} frames")
    return list(targets.values()), driven_bones


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def name_action(armature, name: str) -> None:
    animation_data = armature.animation_data
    if animation_data and animation_data.action:
        animation_data.action.name = name
        animation_data.action.use_fake_user = True


def export_files(armature, args: argparse.Namespace) -> None:
    if args.export_fbx is not None:
        args.export_fbx.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.fbx(
            filepath=str(args.export_fbx),
            use_selection=True,
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            # The source model carries its textures inside the FBX; carry them
            # back out, or the export opens as a grey mannequin elsewhere.
            path_mode="COPY",
            embed_textures=True,
        )
        print(f"[retarget] wrote {args.export_fbx}")
    if args.export_glb is not None:
        args.export_glb.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=str(args.export_glb), export_format="GLB", export_animations=True)
        print(f"[retarget] wrote {args.export_glb}")


def render_preview(output_dir: Path, count: int) -> None:
    """Render a handful of frames from the front, purely as a sanity check."""
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("preview_camera")
    camera_data.lens = 60
    camera = bpy.data.objects.new("preview_camera", camera_data)
    camera.location = Vector((0.0, -2.0, 1.35))
    camera.rotation_euler = (1.5708, 0.0, 0.0)  # level, looking along +Y at the signer
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("preview_light", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("preview_light", light_data)
    light.rotation_euler = (0.9, 0.0, -0.6)
    scene.collection.objects.link(light)

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "PNG"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = scene.frame_end - scene.frame_start + 1
    step = max(1, total // max(1, count))
    for frame in range(scene.frame_start, scene.frame_end + 1, step):
        scene.frame_set(frame)
        scene.render.filepath = str(output_dir / f"preview_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[retarget] preview frames written to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    data = load_landmarks(args.landmarks, args.mirror)
    frame_count = len(data["frames"])
    frame_start = args.start_frame
    frame_end = frame_start + frame_count - 1
    print(f"[retarget] {frame_count} frames from {data.get('source_video', args.landmarks.name)}")

    armature = import_model(args.model)
    bones = build_bone_lookup(armature, need_hands=not args.no_hands)
    setup_scene(data.get("fps", 30.0), frame_start, frame_end)

    helpers, driven_bones = solve(armature, bones, data, args)
    bake_bones(armature, driven_bones, frame_start, frame_end)
    if not args.keep_helpers:
        delete_objects(helpers)
    print(f"[retarget] baked {len(driven_bones)} bones over frames {frame_start}-{frame_end}")

    name_action(armature, f"{Path(data.get('source_video', 'capture')).stem}_mocap")
    bpy.context.scene.frame_set(frame_start)

    args.blend_out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_out))
    print(f"[retarget] wrote {args.blend_out}")

    export_files(armature, args)
    if args.render_preview is not None:
        render_preview(args.render_preview, args.preview_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
