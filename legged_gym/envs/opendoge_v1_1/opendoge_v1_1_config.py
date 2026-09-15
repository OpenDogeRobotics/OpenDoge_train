"""Isaac Gym training configuration for the OpenDog V1.1 URDF."""

from legged_gym.envs.opendoge.opendoge_base_config import (
    OpendogeBaseCfg,
    OpendogeBaseCfgPPO,
)


class OpenDogeV11Cfg(OpendogeBaseCfg):
    """OpenDog V1.1 configuration based on the shared OpenDoge base config."""

    class env(OpendogeBaseCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        episode_length_s = 20

    class terrain(OpendogeBaseCfg.terrain):
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        slope_treshold = 0.75

    class init_state(OpendogeBaseCfg.init_state):
        pos = [0.0, 0.0, 0.158]
        dof_reset_noise_range = 0.1
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.6,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.6,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.6,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.8,
            "RR_calf_joint": 1.6,
        }

    class control(OpendogeBaseCfg.control):
        control_type = "P"
        stiffness = {"joint": 12.0}
        damping = {"joint": 0.5}
        action_scale = 0.25
        decimation = 2

    class commands(OpendogeBaseCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 3.0
        heading_command = False
        command_deadzone = 0.0
        low_speed_values = (0.05, 0.10, 0.20)
        low_speed_fraction = 0.40

        class ranges(OpendogeBaseCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class asset(OpendogeBaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/OpenDogV1.1/urdf/OpenDogV1_1.urdf"
        name = "opendoge_v1_1"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 9.5
        max_linear_velocity = 20.0
        armature = 0.005

    class domain_rand(OpendogeBaseCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        added_mass_range = [-0.15, 0.35]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.3
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        randomize_kp = True
        kp_range = [0.95, 1.05]
        randomize_kd = True
        kd_range = [0.95, 1.05]
        disturbance = False
        disturbance_range = [-1.5, 1.5]
        disturbance_interval = 6
        delay = False

    class rewards(OpendogeBaseCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        tracking_sigma = 0.22
        base_height_target = 0.158
        clearance_height_target = -0.110
        feet_air_time_target = 0.11
        feet_air_time_command_threshold = 0.05
        stand_still_command_threshold = 0.03
        soft_torque_limit = 0.80

        class scales(OpendogeBaseCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.5
            ang_vel_xy = -0.10
            orientation = -2.5
            dof_acc = -2e-6
            joint_power = -2e-5
            base_height = -1.5
            default_pos_linear = -0.05
            diagonal_sync = -0.2
            hip_mirror_symmetry = -0.1
            foot_clearance = -0.0
            action_rate = -0.02
            smoothness = -0.04
            feet_air_time = 1.00
            feet_stumble = -0.0
            feet_drag = -0.0
            stand_still = -2.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            collision = -1.0

    class normalization(OpendogeBaseCfg.normalization):
        class obs_scales(OpendogeBaseCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 1.0

    class noise(OpendogeBaseCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(OpendogeBaseCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(OpendogeBaseCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx(OpendogeBaseCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**24


class OpenDogeV11CfgPPO(OpendogeBaseCfgPPO):
    class algorithm(OpendogeBaseCfgPPO.algorithm):
        entropy_coef = 0.0
        learning_rate = 1e-3
        schedule = "adaptive"

    class runner(OpendogeBaseCfgPPO.runner):
        run_name = "opendoge_v1_1_himloco_v1.0"
        experiment_name = "flat_opendoge_v1_1"
        max_iterations = 9000
        save_interval = 200
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        num_steps_per_env = 48
