"""Gentle OpenDoge recovery task: random fallen pose to stable standing."""

from .opendoge_base_config import OpendogeBaseCfg, OpendogeBaseCfgPPO


class OpendogeGetUpCfg(OpendogeBaseCfg):
    class env(OpendogeBaseCfg.env):
        num_observations = 270
        episode_length_s = 4.0

    class terrain(OpendogeBaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class init_state(OpendogeBaseCfg.init_state):
        pos = [0.0, 0.0, 0.10]
        dof_reset_noise_range = 0.15

    class control(OpendogeBaseCfg.control):
        stiffness = {"joint": 24.0}
        damping = {"joint": 0.8}
        action_scale = 0.25
        decimation = 4

    class commands(OpendogeBaseCfg.commands):
        curriculum = False
        num_commands = 3
        resampling_time = 99999.0
        heading_command = False

        class ranges(OpendogeBaseCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class asset(OpendogeBaseCfg.asset):
        terminate_after_contacts_on = []

    class domain_rand(OpendogeBaseCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-0.05, 0.15]
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        push_robots = False
        disturbance = False
        delay = False

    class noise(OpendogeBaseCfg.noise):
        noise_level = 0.5

    class rewards(OpendogeBaseCfg.rewards):
        base_height_target = 0.185
        only_positive_rewards = False

        class scales(OpendogeBaseCfg.rewards.scales):
            upright = 2.0
            base_height = 2.5
            default_pos = -0.15
            torques = -0.0004
            dof_vel = -0.005
            joint_power = -0.0002
            action_rate = -0.05
            dof_acc = -5e-7
            ang_vel_xy = -0.05
            impact = -1.0
            termination = -5.0
            dof_pos_limits = -1.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            orientation = 0.0
            foot_clearance = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            stand_still = 0.0
            dof_vel_limits = 0.0
            smoothness = 0.0
            diagonal_sync = 0.0
            hip_mirror_symmetry = 0.0
            default_pos_linear = 0.0


class OpendogeGetUpCfgPPO(OpendogeBaseCfgPPO):
    class runner(OpendogeBaseCfgPPO.runner):
        experiment_name = "opendoge_getup"
        run_name = "himloco"
        max_iterations = 3000

