"""Shared observation and control helpers for MuJoCo policy frontends."""

from collections import deque

import numpy as np


def quat_rotate_inverse(q, v):
    """Rotate a world-frame vector into the body frame."""
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Position-derivative torque command."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_policy_input(obs_raw, history_buffer, input_dim, num_obs):
    """Pack one observation for 270-, 64-, or 45-dimensional policies."""
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
    """Build the 45-dimensional Isaac Gym-compatible observation."""
    qj = data.qpos[7:7 + num_actions]
    dqj = data.qvel[6:6 + num_actions]
    quat = data.qpos[3:7]
    if use_gyro_sensor:
        omega = data.sensor("angular-velocity").data.astype(np.float32)
    else:
        omega = quat_rotate_inverse(
            quat, data.qvel[3:6]
        ).astype(np.float32)

    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    proj_gravity = quat_rotate_inverse(quat, gravity_vec)
    return np.concatenate([
        cmd * cmd_scale,
        omega * ang_vel_scale,
        proj_gravity,
        (qj - default_dof_pos) * dof_pos_scale,
        dqj * dof_vel_scale,
        action,
    ], axis=0).astype(np.float32)


def make_history(num_obs, num_one_step_obs):
    """Create a zero-initialized policy history buffer."""
    history_len = max(1, num_obs // num_one_step_obs)
    return deque(
        [np.zeros(num_one_step_obs, dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len,
    )
