# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv, period: float, offset: list[float] | None = None
) -> torch.Tensor:
    """Sinusoidal gait phase observation for each offset.

    Returns ``sin`` and ``cos`` of the normalized phase for every offset, concatenated along the
    last dimension, so a policy can observe where it is within the gait cycle.
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    offset = offset if offset is not None else [0.0]
    out = []
    for off in offset:
        p = (global_phase + off) % 1.0
        out.append(torch.sin(p * torch.pi * 2.0))
        out.append(torch.cos(p * torch.pi * 2.0))
    return torch.stack(out, dim=-1)
