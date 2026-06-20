"""
Shared utilities for OpenDoge MuJoCo Sim2Sim.

Observation building, PD control, quaternion math — shared by all sim2sim frontends.
"""

import numpy as np
from collections import deque


def quat_rotate_inverse(q, v):
    """
    Rotate vector v by the inverse of quaternion q (MuJoCo [w, x, y, z] convention).

    Used to project the gravity vector into the body frame.
    """
    q_w = q[0]
    q_vec = q[1:4]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Position-derivative (PD) torque command."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_policy_input(obs_raw, history_buffer, input_dim, num_obs):
    """
    Pack a single-step observation into the full policy input tensor.

    Supports three ONNX input shapes:
      - num_obs    (e.g. 270): history of concatenated single-step observations
      - 64:         proprietary HIMLoco encoder input
      - 45:         single-step, no history
    """
    if input_dim == num_obs:
        history_buffer.appendleft(obs_raw.copy())
        return np.concatenate(list(history_buffer), axis=0).reshape(1, -1)
    if input_dim == 64:
        policy_input = np.zeros((1, 64), dtype=np.float32)
        policy_input[0, :45] = obs_raw
        return policy_input
    if input_dim == 45:
        return obs_raw.reshape(1, -1)
    raise ValueError(f"Unsupported ONNX input dim: {input_dim}")


def build_obs_raw(data, default_dof_pos, cmd, cmd_scale, ang_vel_scale,
                  dof_pos_scale, dof_vel_scale, action, num_actions,
                  use_gyro_sensor=True):
    """
    Build the single-step observation vector from MuJoCo data.

    Observation order (45 dims):
      Cmd(3) + AngVel(3) + Gravity(3) + DofPos(12) + DofVel(12) + LastAction(12)

    Returns a float32 np.ndarray of shape (45,).
    """
    qj = data.qpos[7:7 + num_actions]
    dqj = data.qvel[6:6 + num_actions]
    quat = data.qpos[3:7]  # [w, x, y, z]

    if use_gyro_sensor:
        omega = data.sensor("angular-velocity").data.astype(np.float32)
    else:
        omega = data.qvel[3:6].astype(np.float32)

    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    proj_gravity = quat_rotate_inverse(quat, gravity_vec)

    qj_norm = (qj - default_dof_pos) * dof_pos_scale
    dqj_norm = dqj * dof_vel_scale
    omega_norm = omega * ang_vel_scale
    cmd_norm = cmd * cmd_scale

    return np.concatenate(
        [cmd_norm, omega_norm, proj_gravity, qj_norm, dqj_norm, action],
        axis=0,
    ).astype(np.float32)
