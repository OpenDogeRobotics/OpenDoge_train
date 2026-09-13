"""OpenDoge recovery environment."""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot


class OpendogeGetUp(LeggedRobot):
    """Train a recovery controller from randomized fallen poses."""

    def _reset_dofs(self, env_ids):
        noise_range = getattr(self.cfg.init_state, "dof_reset_noise_range", 0.15)
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -noise_range, noise_range, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        count = len(env_ids)
        roll = torch_rand_float(-3.0, 3.0, (count, 1), device=self.device).squeeze(1)
        pitch = torch_rand_float(-1.2, 1.2, (count, 1), device=self.device).squeeze(1)
        yaw = torch_rand_float(-3.14, 3.14, (count, 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 2] = torch_rand_float(
            0.08, 0.16, (count, 1), device=self.device
        ).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reward_upright(self):
        return torch.exp(-4.0 * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1))

    def _reward_base_height(self):
        error = self.root_states[:, 2] - self.cfg.rewards.base_height_target
        return torch.exp(-100.0 * torch.square(error))

    def _reward_default_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_impact(self):
        return torch.square(self.base_lin_vel[:, 2]) + 0.25 * torch.sum(
            torch.square(self.base_ang_vel[:, :2]), dim=1
        )
