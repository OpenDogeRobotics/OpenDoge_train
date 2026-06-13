"""MuJoCo 仿真共享工具 — config 加载、模型初始化、PD 控制器装配。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from opendoge_mujoco.position_controller import PDPositionController


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_config(config_path: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (config_path.parent / p).resolve()


def joint_group(joint_name: str) -> str:
    if "_hip_" in joint_name:
        return "hip"
    if "_thigh_" in joint_name:
        return "thigh"
    if "_calf_" in joint_name:
        return "calf"
    raise ValueError(f"Cannot infer gain group for joint: {joint_name}")


def expand_gains(
    joint_names: list[str], gains_config: dict[str, Any]
) -> tuple[dict[str, float], dict[str, float]]:
    kp: dict[str, float] = {}
    kd: dict[str, float] = {}
    for name in joint_names:
        group = joint_group(name)
        kp[name] = float(gains_config[group]["kp"])
        kd[name] = float(gains_config[group]["kd"])
    return kp, kd


def ordered_values(joint_names: list[str], values: dict[str, float]) -> np.ndarray:
    missing = [name for name in joint_names if name not in values]
    if missing:
        raise ValueError(f"Missing joint values for: {', '.join(missing)}")
    return np.array([values[name] for name in joint_names], dtype=np.float64)


def init_sim(
    config_path: Path,
    model_path_override: Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any], list[str], PDPositionController, np.ndarray]:
    """加载 config、初始化 MuJoCo 模型和 PD 控制器，返回全部核心对象。

    Returns:
        model, data, config, joint_names, controller, default_q
    """
    config = load_config(config_path)

    mp = model_path_override.resolve() if model_path_override else resolve_from_config(config_path, config["model_path"])
    model = mujoco.MjModel.from_xml_path(str(mp))
    model.opt.timestep = float(config["control_dt"])
    data = mujoco.MjData(model)

    joint_names = list(config["joint_order"])
    kp, kd = expand_gains(joint_names, config["gains"])
    controller = PDPositionController(model, joint_names, kp, kd)
    default_q = ordered_values(joint_names, config["default_joint_angles"])

    _init_pose(model, data, controller, default_q, config["base_pose"])

    return model, data, config, joint_names, controller, default_q


def _init_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: PDPositionController,
    default_q: np.ndarray,
    base_pose: dict[str, Any],
) -> None:
    mujoco.mj_resetData(model, data)
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "float_base")
    if free_joint_id >= 0:
        qpos_addr = model.jnt_qposadr[free_joint_id]
        data.qpos[qpos_addr : qpos_addr + 3] = np.array(base_pose["position"], dtype=np.float64)
        data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(base_pose["quaternion"], dtype=np.float64)
    data.qpos[controller.qpos_addr] = controller.clamp_to_joint_limits(default_q)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
