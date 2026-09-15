"""Isaac Gym training configuration for the OpenDog V1.1 URDF."""

from legged_gym.envs.opendoge.opendoge_config import OpendogeCfg, OpendogeCfgPPO


class OpenDogeV11Cfg(OpendogeCfg):
    """OpenDoge training setup adapted to the OpenDog V1.1 kinematics."""

    class init_state(OpendogeCfg.init_state):
        # The fixed foot collision sphere has a 15 mm radius; this height puts
        # its bottom on the z=0 plane in the default standing pose.
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

    class asset(OpendogeCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/OpenDogV1.1/urdf/OpenDogV1_1.urdf"
        name = "opendoge_v1_1"

        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]

    class control(OpendogeCfg.control):
        action_scale = 0.25

    class domain_rand(OpendogeCfg.domain_rand):
        # Learn a clean nominal gait first; robustness is added in a later
        # fine-tuning stage instead of being mixed into gait formation.
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        randomize_kp = True
        kp_range = [0.95, 1.05]
        randomize_kd = True
        kd_range = [0.95, 1.05]
        push_robots = False
        disturbance = False
        delay = False

    class commands(OpendogeCfg.commands):
        heading_command = False
        command_deadzone = 0.0
        low_speed_values = (0.05, 0.10, 0.20)
        low_speed_fraction = 0.40

        class ranges(OpendogeCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]

    class rewards(OpendogeCfg.rewards):
        base_height_target = 0.158
        feet_air_time_target = 0.11
        feet_air_time_command_threshold = 0.05
        stand_still_command_threshold = 0.03
        soft_torque_limit = 0.80

        class scales(OpendogeCfg.rewards.scales):
            # V1.1-only penalties are disabled while re-establishing the
            # baseline gait; sparse contact signals distort early updates.
            foot_clearance = -0.0
            feet_stumble = -0.0
            feet_drag = -0.0
            torques = -0.0
            torque_limits = -0.0

    class normalization(OpendogeCfg.normalization):
        clip_actions = 1.0


class OpenDogeV11CfgPPO(OpendogeCfgPPO):
    class algorithm(OpendogeCfgPPO.algorithm):
        # The policy std is trainable and unbounded, so entropy otherwise
        # keeps increasing exploration after the gait starts to stabilize.
        entropy_coef = 0.0

    class runner(OpendogeCfgPPO.runner):
        run_name = "opendoge_v1_1_himloco_v1.0"
        experiment_name = "flat_opendoge_v1_1"
