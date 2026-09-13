"""OpenDoge AMP and HIM-AMP task configurations."""

import glob

from .opendoge_base_config import OpendogeBaseCfg, OpendogeBaseCfgPPO


# Bootstrap clips are carried over from HIMloco and keep the AMPLoader
# contract intact.  For production AMP training, replace these with clips
# retargeted to the OpenDoge URDF and joint-axis conventions.
MOTION_FILES = sorted(glob.glob("datasets/sdog2_motion/*.txt"))


class OpendogeAMPCfg(OpendogeBaseCfg):
    class env(OpendogeBaseCfg.env):
        num_observations = 45
        reference_state_initialization = False
        amp_motion_files = MOTION_FILES

    class terrain(OpendogeBaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class domain_rand(OpendogeBaseCfg.domain_rand):
        push_robots = False
        disturbance = False


class OpendogeAMPCfgPPO(OpendogeBaseCfgPPO):
    runner_class_name = "AMPOnPolicyRunner"

    class runner(OpendogeBaseCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "AMPPPO"
        experiment_name = "opendoge_amp"
        run_name = "himloco"
        max_iterations = 20000
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 200000
        amp_reward_coef = 0.2
        amp_task_reward_lerp = 0.8
        amp_discr_hidden_dims = [512, 256]
        min_normalized_std = [0.01] * 12

    class algorithm(OpendogeBaseCfgPPO.algorithm):
        amp_replay_buffer_size = 200000


class OpendogeHIMAMPCfg(OpendogeBaseCfg):
    class env(OpendogeBaseCfg.env):
        num_observations = 270
        reference_state_initialization = False
        amp_motion_files = MOTION_FILES

    class terrain(OpendogeBaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class domain_rand(OpendogeBaseCfg.domain_rand):
        push_robots = False
        disturbance = False


class OpendogeHIMAMPCfgPPO(OpendogeBaseCfgPPO):
    runner_class_name = "HIMAMPOnPolicyRunner"

    class runner(OpendogeBaseCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMAMPPPO"
        experiment_name = "opendoge_him_amp"
        run_name = "himloco"
        max_iterations = 10000
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 200000
        amp_reward_coef = 0.2
        amp_task_reward_lerp = 0.8
        amp_discr_hidden_dims = [512, 256]
        min_normalized_std = [0.01] * 12

    class algorithm(OpendogeBaseCfgPPO.algorithm):
        amp_replay_buffer_size = 200000
