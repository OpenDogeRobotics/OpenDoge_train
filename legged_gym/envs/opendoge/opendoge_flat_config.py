"""Flat-ground OpenDoge walking task."""

from .opendoge_base_config import OpendogeBaseCfg, OpendogeBaseCfgPPO


class OpendogeFlatCfg(OpendogeBaseCfg):
    class env(OpendogeBaseCfg.env):
        episode_length_s = 20.0

    class terrain(OpendogeBaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True

    class commands(OpendogeBaseCfg.commands):
        curriculum = True
        max_curriculum = 4.0
        num_commands = 4
        resampling_time = 5.0
        heading_command = True

        class ranges(OpendogeBaseCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.5, 1.5]
            ang_vel_yaw = [-2.5, 2.5]
            heading = [-3.14, 3.14]

    class domain_rand(OpendogeBaseCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-0.15, 0.35]
        randomize_com_displacement = False
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]
        push_robots = False
        disturbance = False
        delay = True

    class rewards(OpendogeBaseCfg.rewards):
        base_height_target = 0.185
        tracking_sigma = 0.25

        class scales(OpendogeBaseCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.5
            stand_still = -0.5
            collision = -1.0
            diagonal_sync = -0.1
            hip_mirror_symmetry = -0.1


class OpendogeFlatCfgPPO(OpendogeBaseCfgPPO):
    class runner(OpendogeBaseCfgPPO.runner):
        experiment_name = "opendoge_flat"
        run_name = "himloco"
        max_iterations = 10000

