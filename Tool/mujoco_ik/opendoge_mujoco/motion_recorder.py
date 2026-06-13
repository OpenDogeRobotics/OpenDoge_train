"""MotionRecorder — 将 MuJoCo 仿真帧记录为 .npy 参考运动数据，供 AMP/HIM 训练使用。

输出格式（一个 .npz 文件）：
    joint_positions   (T, 12)  float64  关节位置 [rad]
    joint_velocities  (T, 12)  float64  关节速度 [rad/s]
    joint_torques     (T, 12)  float64  关节力矩 [Nm]
    base_position     (T, 3)   float64  机身世界坐标 [m]
    base_quaternion   (T, 4)   float64  机身朝向 (w, x, y, z)
    base_linear_vel   (T, 3)   float64  机身线速度（世界系）[m/s]
    base_angular_vel  (T, 3)   float64  机身角速度（世界系）[rad/s]
    feet_positions    (T, 4, 3) float64 足端世界坐标 [m]
    body_command      (T, 3)   float64  速度指令 (vx, vy, yaw)
    timestep          float             仿真时间步长 [s]
    joint_names       (12,)    str      关节名称列表

用法：
    recorder = MotionRecorder(joint_names, timestep, output_path)
    for each step:
        recorder.record_frame(data, model, controller, command)
    recorder.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


class MotionRecorder:
    """逐帧记录 MuJoCo 仿真状态，最终保存为 .npz 文件。"""

    def __init__(
        self,
        joint_names: Sequence[str],
        timestep: float,
        output_path: Path | str,
        leg_names: Sequence[str] | None = None,
    ) -> None:
        self.joint_names = list(joint_names)
        self.timestep = float(timestep)
        self.output_path = Path(output_path)
        self.leg_names = list(leg_names) if leg_names else ["FL", "FR", "RL", "RR"]

        self._joint_positions: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._joint_torques: list[np.ndarray] = []
        self._base_positions: list[np.ndarray] = []
        self._base_quaternions: list[np.ndarray] = []
        self._base_linear_vels: list[np.ndarray] = []
        self._base_angular_vels: list[np.ndarray] = []
        self._feet_positions: list[np.ndarray] = []
        self._body_commands: list[np.ndarray] = []

    def record_frame(
        self,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        q_des: np.ndarray,
        tau: np.ndarray,
        body_command: np.ndarray | None = None,
    ) -> None:
        """记录一帧仿真状态。

        Args:
            data: MuJoCo MjData（已执行过 mj_step 或 mj_forward）
            model: MuJoCo MjModel
            q_des: 当前帧的期望关节位置 (12,)
            tau: 当前帧的实际关节力矩 (12,)
            body_command: 速度指令 (vx, vy, yaw)，None 则为零指令
        """
        # 关节状态
        q = q_des.copy()
        dq = data.qvel.copy()[: len(self.joint_names)]  # simplified — 取前 12 个速度自由度
        self._joint_positions.append(q)
        self._joint_velocities.append(dq)
        self._joint_torques.append(tau.copy() if tau is not None else np.zeros(len(self.joint_names)))

        # 机身状态 — 从 float_base joint 读取
        free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "float_base")
        if free_joint_id >= 0:
            qpos_addr = model.jnt_qposadr[free_joint_id]
            qvel_addr = model.jnt_dofadr[free_joint_id]
            self._base_positions.append(data.qpos[qpos_addr : qpos_addr + 3].copy())
            self._base_quaternions.append(data.qpos[qpos_addr + 3 : qpos_addr + 7].copy())
            self._base_linear_vels.append(data.qvel[qvel_addr : qvel_addr + 3].copy())
            self._base_angular_vels.append(data.qvel[qvel_addr + 3 : qvel_addr + 6].copy())
        else:
            self._base_positions.append(np.zeros(3))
            self._base_quaternions.append(np.array([1.0, 0.0, 0.0, 0.0]))
            self._base_linear_vels.append(np.zeros(3))
            self._base_angular_vels.append(np.zeros(3))

        # 足端世界坐标 — 从 body xpos 读取
        feet = np.zeros((len(self.leg_names), 3), dtype=np.float64)
        for i, leg in enumerate(self.leg_names):
            foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            if foot_body_id >= 0:
                feet[i] = data.xpos[foot_body_id].copy()
        self._feet_positions.append(feet)

        # 速度指令
        if body_command is not None:
            self._body_commands.append(np.asarray(body_command, dtype=np.float64).ravel()[:3])
        else:
            self._body_commands.append(np.zeros(3, dtype=np.float64))

    @property
    def num_frames(self) -> int:
        return len(self._joint_positions)

    def save(self) -> Path:
        """将所有记录的帧保存为 .npz 文件，返回输出路径。"""
        if self.num_frames == 0:
            raise RuntimeError("No frames recorded — call record_frame() before save().")

        T = self.num_frames
        N = len(self.joint_names)
        F = len(self.leg_names)

        np.savez_compressed(
            self.output_path,
            joint_positions=np.array(self._joint_positions, dtype=np.float64).reshape(T, N),
            joint_velocities=np.array(self._joint_velocities, dtype=np.float64).reshape(T, N),
            joint_torques=np.array(self._joint_torques, dtype=np.float64).reshape(T, N),
            base_position=np.array(self._base_positions, dtype=np.float64).reshape(T, 3),
            base_quaternion=np.array(self._base_quaternions, dtype=np.float64).reshape(T, 4),
            base_linear_vel=np.array(self._base_linear_vels, dtype=np.float64).reshape(T, 3),
            base_angular_vel=np.array(self._base_angular_vels, dtype=np.float64).reshape(T, 3),
            feet_positions=np.array(self._feet_positions, dtype=np.float64).reshape(T, F, 3),
            body_command=np.array(self._body_commands, dtype=np.float64).reshape(T, 3),
            timestep=np.float64(self.timestep),
            joint_names=np.array(self.joint_names, dtype=str),
            leg_names=np.array(self.leg_names, dtype=str),
        )
        return self.output_path
