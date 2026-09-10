"""Isaac Gym training configuration for the OpenDog V1.1 URDF."""

from legged_gym.envs.opendoge.opendoge_config import OpendogeCfg, OpendogeCfgPPO


class OpenDogeV11Cfg(OpendogeCfg):
    """OpenDoge training setup adapted to the OpenDog V1.1 kinematics."""

    class init_state(OpendogeCfg.init_state):
        # V1.1 has shorter leg links and no separate foot link.
        pos = [0.0, 0.0, 0.15]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.6,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.6,
            "FR_calf_joint": 1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.6,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.6,
            "RR_calf_joint": 1.5,
        }

    class asset(OpendogeCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/OpenDogV1.1/urdf/OpenDog.SLDASM.urdf"
        name = "opendoge_v1_1"

        # V1.1 ends at the calf; use calf bodies as the four foot contacts.
        foot_name = "calf"
        penalize_contacts_on = ["hip", "thigh", "base"]
        terminate_after_contacts_on = ["base"]

    class rewards(OpendogeCfg.rewards):
        base_height_target = 0.15


class OpenDogeV11CfgPPO(OpendogeCfgPPO):
    class runner(OpendogeCfgPPO.runner):
        run_name = "opendoge_v1_1_himloco_v1.0"
        experiment_name = "flat_opendoge_v1_1"
