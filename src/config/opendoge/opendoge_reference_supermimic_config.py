# Copyright (c) 2026-2027 zh
"""内容：OpenDoge 参考任务的 SuperMimic 配置；继承 Go2 协议并按 URDF 尺度适配。

OpenDoge（OpenDog V1.1）站高约 0.158 m，约为 Go2（0.32 m）的一半尺度。
控制器参数取自 OpenDoge 资产标称值；速度课程、地形难度、扫描范围与
步态接触高度按整机尺度约 0.55 倍缩放。
"""

import math

from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg


class OpendogeReferenceCfg(Go2ReferenceCfg):
    """OpenDoge 的新训练默认；任务协议与奖励结构与 Go2 参考保持一致。"""

    class asset(Go2ReferenceCfg.asset):
        robot = "opendoge_reference"
        base_body = "base_link"

    class control(Go2ReferenceCfg.control):
        """OpenDoge 标称位置执行器；动作比例按关节类型设置。"""

        stiffness = {"hip": 12.0, "thigh": 12.0, "calf": 12.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        effort_limit = {"hip": 6.0, "thigh": 6.0, "calf": 9.0}
        velocity_limit = {"hip": 30.1, "thigh": 30.1, "calf": 20.07}
        armature = {"hip": 0.005, "thigh": 0.005, "calf": 0.005}
        action_scales = {"hip": 0.25, "thigh": 0.4, "calf": 0.4}

    class init_state(Go2ReferenceCfg.init_state):
        """OpenDoge 出生位姿与零动作对应的关节目标（FR/RR 腿为镜像符号）。"""

        pos = [0.0, 0.0, 0.158]
        default_joint = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "FR_thigh_joint": -0.8,
            "RR_thigh_joint": -0.8,
            "FL_calf_joint": -1.6,
            "RL_calf_joint": -1.6,
            "FR_calf_joint": 1.6,
            "RR_calf_joint": 1.6,
        }

    class commands(Go2ReferenceCfg.commands):
        """速度课程按 0.55 倍整机尺度缩放。"""

        initial_ranges = ((-0.6, 0.6), (-0.5, 0.5), (-1.5, 1.5))

    class terrain(Go2ReferenceCfg.terrain):
        """地形难度按 URDF 腿长比例缩放；扫描范围同步缩小。"""

        scan_size_m = (0.72, 0.432)
        scan_resolution_m = 0.072
        stair_dimensions = ((0.10, 1.2), (0.11, 1.1), (0.12, 1.05), (0.14, 0.95))
        stair_height_range = (0.03, 0.10)
        box_grid_height_range = (0.01, 0.055)

    class domain_randomization(Go2ReferenceCfg.domain_randomization):
        """质心扰动按基座尺寸缩放；摩擦与编码器偏差沿用参考。"""

        com_range = {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.018, 0.018)}

    class reward(Go2ReferenceCfg.reward):
        class gait(Go2ReferenceCfg.reward.gait):
            """小尺度机器人步态频率略高；接触阈值按足端半径 0.015 m 调整。"""

            period = 0.5
            contact_height = 0.02

        class posture(Go2ReferenceCfg.reward.posture):
            """姿态标准差按关节行程缩放。"""

            std_standing = {"hip": 0.05, "thigh": 0.1, "calf": 0.15}
            std_walking = {"hip": 0.12, "thigh": 0.28, "calf": 0.4}
            std_running = {"hip": 0.12, "thigh": 0.28, "calf": 0.4}

    class viewer(Go2ReferenceCfg.viewer):
        """OpenDoge 跟随相机。"""

        distance = 1.0
