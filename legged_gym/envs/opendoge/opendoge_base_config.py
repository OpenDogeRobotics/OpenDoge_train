"""Shared OpenDoge configuration used by the task-specific training setups.

The legacy ``OpendogeCfg`` remains the backwards-compatible default.  New
tasks inherit through this module so their terrain, reset distribution, reward
weights, and runner names are isolated from one another.
"""

from .opendoge_config import OpendogeCfg, OpendogeCfgPPO


class OpendogeBaseCfg(OpendogeCfg):
    """Common OpenDoge kinematics, observations, and simulator settings."""


class OpendogeBaseCfgPPO(OpendogeCfgPPO):
    """Common PPO/HIM defaults for OpenDoge tasks."""

    class runner(OpendogeCfgPPO.runner):
        save_interval = 200

