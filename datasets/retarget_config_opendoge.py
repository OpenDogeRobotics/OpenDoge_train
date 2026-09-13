"""Retargeting configuration for OpenDoge AMP clips."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
URDF_FILENAME = str(ROOT / "resources/robots/Opendoge/urdf/Opendoge.urdf")
OUTPUT_DIR = str(ROOT / "datasets/opendoge_motion")

# The source dog clips have approximately 0.40 m leg segments. OpenDoge has
# approximately 0.30 m from hip to foot, so scale the reference skeleton to
# the target kinematics before IK.
REF_POS_SCALE = 0.36
FRAME_DURATION = 0.01677

INIT_POS = np.array([0.0, 0.0, 0.15], dtype=np.float64)
INIT_ROT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# OpenDoge URDF / policy order.
URDF_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
POLICY_JOINT_ORDER = list(URDF_JOINT_ORDER)

# PyBullet link indices with URDF_MAINTAIN_LINK_ORDER enabled.
SIM_HIP_JOINT_IDS = [4, 0, 12, 8]
SIM_TOE_JOINT_IDS = [7, 3, 15, 11]
SIM_TOE_JOINT_IDS_POLICY = [3, 7, 11, 15]
# Slices in PyBullet's actuated-joint order for source order FR, FL, RR, RL.
SIM_IK_SEGMENTS = [(3, 6), (0, 3), (9, 12), (6, 9)]

# Reference keypoint indices: pelvis, neck, four shoulders, four hips, four toes.
REF_PELVIS_JOINT_ID = 0
REF_NECK_JOINT_ID = 3
REF_HIP_JOINT_IDS = [6, 11, 16, 20]  # left/right shoulder, left/right hip
REF_TOE_JOINT_IDS = [10, 15, 19, 23]  # left/right front, left/right rear

# Coordinate conversion used by the original AI4Animation clips.
REF_COORD_ROT = np.array([0.70710678, 0.0, 0.0, 0.70710678])
REF_ROOT_ROT = np.array([0.0, 0.0, 0.67249851, 0.74008418])
FORWARD_DIR_OFFSET = np.zeros(3)

# OpenDoge default standing pose, in the explicit policy/URDF order.
DEFAULT_JOINT_POSE = np.array([
    0.0, 0.6, -1.5,
    0.0, 0.6, -1.5,
    0.0, 0.6, -1.5,
    0.0, 0.6, -1.5,
], dtype=np.float64)

JOINT_DAMPING = [0.10, 0.05, 0.01] * 4
SIM_ROOT_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
TOE_HEIGHT_OFFSET = 0.012

MOCAP_MOTIONS = [
    ["pace0", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 162, 201, 1.0],
    ["pace1", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 201, 400, 1.0],
    ["pace2", "datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 400, 600, 1.0],
    ["trot0", "datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 448, 481, 1.0],
    ["trot1", "datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 400, 600, 1.0],
    ["trot2", "datasets/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt", 480, 663, 1.0],
    ["canter0", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 430, 480, 1.0],
    ["canter1", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 380, 430, 1.0],
    ["canter2", "datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 480, 566, 1.0],
    ["right_turn0", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 1085, 1124, 1.5],
    ["right_turn1", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 560, 670, 1.5],
    ["left_turn0", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 2404, 2450, 1.5],
    ["left_turn1", "datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 120, 220, 1.5],
]
