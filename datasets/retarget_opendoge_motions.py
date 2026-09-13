"""Retarget AI4Animation dog keypoints to the OpenDoge URDF.

The generated files use the AMP JSON schema consumed by
``rsl_rl.datasets.motion_loader.AMPLoader``:

    root_pos(3), root_quat(4), joint_pos(12), feet_pos_local(12),
    base_lin_vel(3), base_ang_vel(3), joint_vel(12), feet_vel_local(12)

Run from the repository root:

    python datasets/retarget_opendoge_motions.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

import retarget_config_opendoge as config


POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
FEET_POS_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
FEET_VEL_SIZE = 12


def quat_multiply(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], dtype=np.float64)


def quat_inverse(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / np.dot(q, q)


def quat_rotate(vector, quaternion):
    vector_quat = np.array([vector[0], vector[1], vector[2], 0.0])
    return quat_multiply(quat_multiply(quaternion, vector_quat), quat_inverse(quaternion))[:3]


def quat_about_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def transform_reference_points(points):
    transformed = np.asarray(points, dtype=np.float64).copy()
    for index in range(transformed.shape[0]):
        transformed[index] = quat_rotate(transformed[index], config.REF_COORD_ROT)
        transformed[index] = quat_rotate(transformed[index], config.REF_ROOT_ROT)
        transformed[index] *= config.REF_POS_SCALE
    return transformed


def load_reference(path, start, end):
    values = np.loadtxt(path, delimiter=",")
    values = values[start: end + 1]
    if values.shape[1] % 3 != 0:
        raise ValueError(f"Expected xyz keypoints in {path}, got {values.shape}")
    return values.reshape(values.shape[0], -1, 3)


def retarget_root_pose(reference):
    pelvis = reference[config.REF_PELVIS_JOINT_ID]
    neck = reference[config.REF_NECK_JOINT_ID]
    left_shoulder = reference[config.REF_HIP_JOINT_IDS[0]]
    right_shoulder = reference[config.REF_HIP_JOINT_IDS[1]]
    left_hip = reference[config.REF_HIP_JOINT_IDS[2]]
    right_hip = reference[config.REF_HIP_JOINT_IDS[3]]

    forward = neck - pelvis + config.FORWARD_DIR_OFFSET
    forward /= max(np.linalg.norm(forward), 1e-8)
    shoulder = left_shoulder - right_shoulder
    hip = left_hip - right_hip
    shoulder /= max(np.linalg.norm(shoulder), 1e-8)
    hip /= max(np.linalg.norm(hip), 1e-8)
    left = 0.5 * (shoulder + hip)
    left /= max(np.linalg.norm(left), 1e-8)
    up = np.cross(forward, left)
    up /= max(np.linalg.norm(up), 1e-8)
    left = np.cross(up, forward)
    left[2] = 0.0
    left /= max(np.linalg.norm(left), 1e-8)

    rotation = np.array([
        [forward[0], left[0], up[0]],
        [forward[1], left[1], up[1]],
        [forward[2], left[2], up[2]],
    ])
    # Convert the orthonormal matrix to a quaternion through PyBullet.
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        root_rot = np.array([
            (rotation[2, 1] - rotation[1, 2]) * s,
            (rotation[0, 2] - rotation[2, 0]) * s,
            (rotation[1, 0] - rotation[0, 1]) * s,
            0.25 / s,
        ])
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        q = np.zeros(4)
        s = np.sqrt(1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index]) * 2.0
        q[index] = 0.25 * s
        q[3] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / s
        q[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / s
        q[last_index] = (rotation[last_index, index] + rotation[index, last_index]) / s
        root_rot = q
    root_rot = quat_multiply(root_rot, config.INIT_ROT)
    root_rot /= max(np.linalg.norm(root_rot), 1e-8)

    root_pos = 0.5 * (pelvis + neck)
    return root_pos + config.SIM_ROOT_OFFSET, root_rot


def foot_positions_local(robot, root_pos, root_rot):
    inverse_root_pos, inverse_root_rot = p.invertTransform(root_pos.tolist(), root_rot.tolist())
    feet = []
    for link_id in config.SIM_TOE_JOINT_IDS_POLICY:
        link_state = p.getLinkState(robot, link_id, computeForwardKinematics=True)
        local_pos, _ = p.multiplyTransforms(
            inverse_root_pos, inverse_root_rot, link_state[4], link_state[5]
        )
        feet.extend(local_pos)
    return np.asarray(feet, dtype=np.float64)


def retarget_pose(robot, reference):
    root_pos, root_rot = retarget_root_pose(reference)
    p.resetBasePositionAndOrientation(robot, root_pos.tolist(), root_rot.tolist())

    forward = quat_rotate([1.0, 0.0, 0.0], root_rot)
    heading_rot = quat_about_axis(
        np.arctan2(forward[1], forward[0]), [0.0, 0.0, 1.0]
    )
    targets = []
    for toe_index, hip_index, sim_hip_id in zip(
        config.REF_TOE_JOINT_IDS,
        config.REF_HIP_JOINT_IDS,
        config.SIM_HIP_JOINT_IDS,
    ):
        hip_state = p.getLinkState(robot, sim_hip_id, computeForwardKinematics=True)
        target = np.asarray(hip_state[4]) + quat_rotate(
            reference[toe_index] - reference[hip_index], heading_rot
        )
        target[2] = reference[toe_index, 2]
        target[2] += config.TOE_HEIGHT_OFFSET
        targets.append(target.tolist())

    lower = [
        -0.785, -0.785, -2.68,
        -0.26, -0.785, -2.68,
        -0.785, -0.785, -2.68,
        -0.26, -0.785, -2.68,
    ]
    upper = [
        0.26, 1.134, -1.04,
        0.785, 1.134, -1.04,
        0.26, 1.134, -1.04,
        0.785, 1.134, -1.04,
    ]
    joint_ranges = [upper_value - lower_value for lower_value, upper_value in zip(lower, upper)]
    # A multi-end-effector solve converges to a symmetric compromise on this
    # URDF. Solve each leg independently so the source gait is preserved.
    joint_pose = np.zeros(12, dtype=np.float64)
    for target, toe_link, (start, end) in zip(
        targets, config.SIM_TOE_JOINT_IDS, config.SIM_IK_SEGMENTS
    ):
        leg_pose = np.asarray(
            p.calculateInverseKinematics(
                robot,
                toe_link,
                target,
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=joint_ranges,
                jointDamping=config.JOINT_DAMPING,
                restPoses=config.DEFAULT_JOINT_POSE.tolist(),
                maxNumIterations=100,
                residualThreshold=1e-5,
            ),
            dtype=np.float64,
        )
        if leg_pose.size != 12:
            raise RuntimeError(f"OpenDoge leg IK returned {leg_pose.size} joints")
        joint_pose[start:end] = leg_pose[start:end]
    joint_pose = np.clip(joint_pose, lower, upper)
    # The fixed joints are ignored by PyBullet; reset only the 12 actuated DOFs.
    actuated_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    for joint_id, angle in zip(actuated_ids, joint_pose):
        p.resetJointState(robot, joint_id, float(angle), 0.0)
    p.resetBasePositionAndOrientation(robot, root_pos.tolist(), root_rot.tolist())
    feet_local = foot_positions_local(robot, root_pos, root_rot)
    return np.concatenate([root_pos, root_rot, joint_pose, feet_local])


def retarget_motion(robot, references):
    poses = [retarget_pose(robot, frame) for frame in references]
    poses = np.asarray(poses)
    frame_count = poses.shape[0] - 1
    frames = np.zeros((frame_count, 61), dtype=np.float64)
    dt = config.FRAME_DURATION
    frames[:, :31] = poses[:-1, :31]

    for i in range(frame_count):
        current = poses[i]
        following = poses[i + 1]
        root_pos, root_rot = current[:3], current[3:7]
        inverse_root_rot = quat_inverse(root_rot)
        frames[i, 31:34] = quat_rotate((following[:3] - root_pos) / dt, inverse_root_rot)
        diff_quat = p.getDifferenceQuaternion(root_rot.tolist(), following[3:7].tolist())
        axis, angle = p.getAxisAngleFromQuaternion(diff_quat)
        frames[i, 34:37] = quat_rotate(np.asarray(axis) * angle / dt, inverse_root_rot)
        frames[i, 37:49] = (following[7:19] - current[7:19]) / dt
        frames[i, 49:61] = (following[19:31] - current[19:31]) / dt

    frames[:, :2] -= frames[0, :2]
    return frames


def write_motion(frames, output_path, motion_weight):
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": config.FRAME_DURATION,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": motion_weight,
        "Frames": np.round(frames, 5).tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main(selected=None):
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0.0, 0.0, 0.0)
    robot = p.loadURDF(
        config.URDF_FILENAME,
        config.INIT_POS.tolist(),
        config.INIT_ROT.tolist(),
        flags=p.URDF_MAINTAIN_LINK_ORDER,
        useFixedBase=False,
    )
    output_dir = Path(config.OUTPUT_DIR)
    motions = config.MOCAP_MOTIONS if selected is None else [m for m in config.MOCAP_MOTIONS if m[0] in selected]
    for name, relative_path, start, end, weight in motions:
        source = Path(relative_path)
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[1] / source
        references = load_reference(source, start, end)
        references = np.asarray([transform_reference_points(frame) for frame in references])
        reference_root_height = 0.5 * (
            references[0, config.REF_PELVIS_JOINT_ID, 2]
            + references[0, config.REF_NECK_JOINT_ID, 2]
        )
        references[:, :, 2] += config.INIT_POS[2] - reference_root_height
        for toe_index in config.REF_TOE_JOINT_IDS:
            references[:, toe_index, 2] += config.TOE_HEIGHT_OFFSET
        frames = retarget_motion(robot, references)
        output_path = output_dir / f"{name}.json"
        write_motion(frames, output_path, weight)
        print(f"wrote {output_path} frames={len(frames)} shape={frames.shape}")
    p.disconnect(physics_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", action="append", help="Only generate this named clip")
    args = parser.parse_args()
    main(args.motion)
