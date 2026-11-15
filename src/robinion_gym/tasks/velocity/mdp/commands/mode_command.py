# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitModeCommand(CommandTerm):
    """Binary command that selects walking or standing still.

    The command is ``1`` in walk mode and ``0`` in stand mode. Walk mode only enables the gait;
    the velocity command decides whether the robot translates. A zero velocity command in walk
    mode therefore means walking in place, while stand mode forces the velocity command to zero
    and holds the init pose. The command manager resamples the value on episode reset; set a
    resampling interval longer than the episode to keep one mode for the whole episode, or a
    shorter interval to train live stand-to-walk transitions.
    """

    cfg: GaitModeCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: GaitModeCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The gait mode, shape ``(num_envs, 1)``: 0 = stand, 1 = walk."""
        return self._command

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        """Sample the stand/walk mode for the specified environments."""
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return
        walk = torch.rand(num_envs, device=self.device) < self.cfg.walk_probability
        self._command[env_ids] = walk.float().unsqueeze(-1)


@configclass
class GaitModeCommandCfg(CommandTermCfg):
    """Configuration for the gait-mode command generator."""

    class_type: type[GaitModeCommand] = GaitModeCommand

    walk_probability: float = 0.5
    """Probability that an environment is commanded to walk instead of standing still."""
