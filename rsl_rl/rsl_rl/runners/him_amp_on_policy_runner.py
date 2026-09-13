import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.modules import HIMActorCritic
from rsl_rl.algorithms.him_amp_ppo import HIMAMPPPO
from rsl_rl.algorithms.amp_D import AMP_Discriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.env import VecEnv


class HIMAMPOnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic = HIMActorCritic(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg["amp_num_preload_transitions"],
            motion_files=self.cfg["amp_motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMP_Discriminator(
            amp_data.observation_dim * 2,
            self.cfg["amp_reward_coef"],
            self.cfg["amp_discr_hidden_dims"],
            device,
            self.cfg["amp_task_reward_lerp"],
        ).to(self.device)

        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device)
            * (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]))
        )

        amp_replay_buffer_size = self.alg_cfg.pop("amp_replay_buffer_size", 1000000)
        self.alg = HIMAMPPPO(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            amp_replay_buffer_size=amp_replay_buffer_size,
            amp_reward_coef=self.cfg["amp_reward_coef"],
            amp_task_reward_lerp=self.cfg["amp_task_reward_lerp"],
            min_std=min_std,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        amp_obs = self.env.get_amp_observations().to(self.device)

        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    (obs, privileged_obs, rewards, dones, infos,
                     termination_ids, termination_privileged_obs,
                     terminal_amp_states) = self.env.step(actions)

                    critic_obs = (
                        privileged_obs if privileged_obs is not None else obs
                    )
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = (
                        termination_privileged_obs.to(self.device)
                    )
                    terminal_amp_states = terminal_amp_states.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = (
                        termination_privileged_obs.clone().detach()
                    )

                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[termination_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )[0]
                    amp_obs = torch.clone(next_amp_obs)

                    self.alg.process_env_step(
                        rewards, dones, infos, next_critic_obs,
                        next_amp_obs_with_term,
                    )

                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (mean_value_loss, mean_surrogate_loss, mean_estimation_loss,
             mean_swap_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_policy_pred, mean_expert_pred) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(
            self.log_dir,
            'model_{}.pt'.format(self.current_learning_iteration),
        ))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / (locs['collection_time'] + locs['learn_time'])
        )

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        rew_mean = (
            statistics.mean(locs['rewbuffer'])
            if len(locs['rewbuffer']) > 0 else 0.0
        )
        len_mean = (
            statistics.mean(locs['lenbuffer'])
            if len(locs['lenbuffer']) > 0 else 0.0
        )
        str_ = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str_.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
            f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
            f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
            f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            f"""{'Mean reward:':>{pad}} {rew_mean:.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {len_mean:.2f}\n"""
        )
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': (
                self.alg.actor_critic.estimator.optimizer.state_dict()
            ),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(
                loaded_dict['estimator_optimizer_state_dict']
            )
            self.alg.discriminator.load_state_dict(
                loaded_dict['discriminator_state_dict']
            )
            self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
