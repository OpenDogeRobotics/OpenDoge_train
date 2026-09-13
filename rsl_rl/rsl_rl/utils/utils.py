# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np
from typing import Tuple


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batched torch quaternion SLERP used when interpolating AMP clips."""
    if not torch.is_tensor(q0):
        q0 = torch.as_tensor(q0, dtype=torch.float32)
    if not torch.is_tensor(q1):
        q1 = torch.as_tensor(q1, dtype=torch.float32)
    if not torch.is_tensor(fraction):
        fraction = torch.as_tensor(fraction, dtype=torch.float32)
    q0 = q0 / torch.linalg.vector_norm(q0, dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / torch.linalg.vector_norm(q1, dim=-1, keepdim=True).clamp_min(1e-12)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    if shortestpath:
        neg = dot < 0.0
        q1 = torch.where(neg, -q1, q1)
        dot = torch.where(neg, -dot, dot)
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    fraction = fraction.to(q0.dtype)
    while fraction.dim() < q0.dim():
        fraction = fraction.unsqueeze(-1)
    fraction = fraction.expand_as(q0[..., :1])
    slerp = (
        q0 * torch.sin((1.0 - fraction) * theta)
        + q1 * torch.sin(fraction * theta)
    ) / sin_theta.clamp_min(1e-8)
    linear = q0 * (1.0 - fraction) + q1 * fraction
    return torch.where(sin_theta.abs() < 1e-8, linear, slerp)

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


class RunningMeanStd:
    """Streaming statistics used to normalize AMP discriminator inputs."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        self.update_from_moments(batch_mean, batch_var, values.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m_2 / total_count
        self.count = total_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, values):
        return np.clip(
            (values - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize_torch(self, values, device):
        mean = torch.as_tensor(self.mean, device=device, dtype=torch.float32)
        std = torch.sqrt(torch.as_tensor(self.var + self.epsilon, device=device))
        return torch.clamp((values - mean) / std, -self.clip_obs, self.clip_obs)
