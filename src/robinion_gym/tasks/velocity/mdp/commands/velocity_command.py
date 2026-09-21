# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from .commands_cfg import UniformLevelVelocityCommandCfg


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator with a curriculum that ramps velocity ranges up to ``limit_ranges``.

    The command is sampled uniformly within the ranges interpolated between ``ranges`` (level 0)
    and ``limit_ranges`` (level 1). After each command period, the level of each environment is
    increased when the robot tracked the commanded velocity well enough (within ``good_err_threshold``)
    for more than 80% of the steps, and decreased otherwise.
    """

    cfg: UniformLevelVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # curriculum level per environment, in [0, 1]
        self.level = torch.zeros(self.num_envs, device=self.device)
        # per-command-period tracking counters used to advance/retreat the level
        self._num_good_steps = torch.zeros(self.num_envs, device=self.device)
        self._num_steps = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        # track how often the robot follows the commanded linear velocity
        err = torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        self._num_good_steps += (err < self.cfg.good_err_threshold).float()
        self._num_steps += 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        # update the level based on the previous command period
        num_steps = self._num_steps[env_ids]
        good_ratio = self._num_good_steps[env_ids] / num_steps.clamp(min=1.0)
        delta = torch.where(good_ratio > 0.8, self.cfg.level_step, -self.cfg.level_step)
        delta = torch.where(num_steps <= 0, 0.0, delta)
        self.level[env_ids] = torch.clip(self.level[env_ids] + delta, min=0.0, max=1.0)
        # reset the counters for the next command period
        self._num_good_steps[env_ids] = 0.0
        self._num_steps[env_ids] = 0.0

        # sample velocity commands within the interpolated ranges
        lv = self.level[env_ids].unsqueeze(-1)

        def interpolate(start, end):
            s = torch.tensor(start, device=self.device, dtype=torch.float)
            e = torch.tensor(end, device=self.device, dtype=torch.float)
            return s + (e - s) * lv

        lin_x = interpolate(self.cfg.ranges.lin_vel_x, self.cfg.limit_ranges.lin_vel_x)
        lin_y = interpolate(self.cfg.ranges.lin_vel_y, self.cfg.limit_ranges.lin_vel_y)
        ang_z = interpolate(self.cfg.ranges.ang_vel_z, self.cfg.limit_ranges.ang_vel_z)

        lo = torch.stack([lin_x[:, 0], lin_y[:, 0], ang_z[:, 0]], dim=-1)
        hi = torch.stack([lin_x[:, 1], lin_y[:, 1], ang_z[:, 1]], dim=-1)
        self.vel_command_b[env_ids] = lo + torch.rand(len(env_ids), 3, device=self.device) * (hi - lo)

        # heading target (if enabled)
        if self.cfg.heading_command:
            r = torch.rand(len(env_ids), device=self.device)
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # standing envs
        self.is_standing_env[env_ids] = torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_standing_envs

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._num_good_steps[env_ids] = 0.0
        self._num_steps[env_ids] = 0.0
        return super().reset(env_ids)
