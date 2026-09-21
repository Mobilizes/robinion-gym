# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .velocity_command import UniformLevelVelocityCommand


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the level-based uniform velocity command generator."""

    class_type: type["UniformLevelVelocityCommand"] | str = "{DIR}.velocity_command:UniformLevelVelocityCommand"

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The velocity command ranges at the maximum level (level 1)."""

    good_err_threshold: float = 0.15
    """The linear velocity tracking error (in m/s) below which tracking is considered good."""

    level_step: float = 0.05
    """The amount by which the level changes after each command period."""
