"""AMP environment adapter for the legacy OpenDoge simulator."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.legged_robot import LeggedRobot


class OpendogeAMP(LeggedRobot):
    """Expose the HIMloco AMP transition contract on OpenDoge."""

    def __init__(self, *args, **kwargs):
        self._terminal_amp_states = None
        super().__init__(*args, **kwargs)

    def compute_termination_observations(self, env_ids):
        self._terminal_amp_states = self.get_amp_observations()[env_ids].clone()
        return super().compute_termination_observations(env_ids)

    def step(self, actions):
        obs, privileged_obs, rewards, dones, infos, reset_ids, _ = super().step(actions)
        return obs, privileged_obs, rewards, dones, infos, reset_ids, self._terminal_amp_states

    def get_amp_observations(self):
        """Return 43-D AMP state matching AMPLoader.feed_forward_generator."""
        feet_relative = self.feet_pos - self.root_states[:, None, :3]
        feet_local = quat_rotate_inverse(
            self.base_quat[:, None, :].expand_as(feet_relative), feet_relative
        ).reshape(self.num_envs, -1)
        return torch.cat(
            (
                self.dof_pos,
                feet_local,
                self.base_lin_vel,
                self.base_ang_vel,
                self.dof_vel,
                self.root_states[:, 2:3],
            ),
            dim=-1,
        )

