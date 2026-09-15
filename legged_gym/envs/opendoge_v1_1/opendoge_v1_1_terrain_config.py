"""Terrain fine-tuning configuration for the OpenDog V1.1 policy."""

from .opendoge_v1_1_config import OpenDogeV11Cfg, OpenDogeV11CfgPPO


class OpenDogeV11TerrainCfg(OpenDogeV11Cfg):
    """Start on easy terrain and ramp difficulty only after stable walking."""

    class env(OpenDogeV11Cfg.env):
        num_envs = 4096

    class terrain(OpenDogeV11Cfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        max_init_terrain_level = 0
        num_rows = 10
        num_cols = 20
        # The robot has roughly 0.24 m legs. Keep early terrain features in
        # the centimeter range and make curriculum progression gentle.
        difficulty_scale = 1.0
        terrain_height_min = 0.01
        terrain_height_max = 0.10
        slope_scale = 0.20
        gap_scale = 0.20
        pit_scale = 0.20
        stone_size = 0.80
        stone_distance = 0.04
        # Keep the first phase to slopes, mild roughness, stairs and blocks.
        # Zero-weight padding also exercises all generator branches safely.
        terrain_proportions = [0.15, 0.25, 0.35, 0.25, 0.0, 0.0, 0.0]

    class commands(OpenDogeV11Cfg.commands):
        curriculum = True
        max_curriculum = 1.0
        resampling_time = 4.0
        heading_command = False

        class ranges(OpenDogeV11Cfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]

    class control(OpenDogeV11Cfg.control):
        # A small reduction limits impact spikes while retaining the trained
        # action contract and 100 Hz policy update rate.
        stiffness = {"joint": 10.0}
        damping = {"joint": 0.6}
        action_scale = 0.27

    class rewards(OpenDogeV11Cfg.rewards):
        tracking_sigma = 0.28

        class scales(OpenDogeV11Cfg.rewards.scales):
            torques = -1.0e-3
            torque_limits = -0.05
            dof_vel = -1.0e-4
            action_rate = -0.03
            smoothness = -0.05
            collision = -1.5


class OpenDogeV11TerrainCfgPPO(OpenDogeV11CfgPPO):
    class runner(OpenDogeV11CfgPPO.runner):
        run_name = "opendoge_v1_1_terrain_finetune"
        experiment_name = "flat_opendoge_v1_1"
        save_interval = 100
