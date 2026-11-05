# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def leg_leg_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when a leg body touches the opposite leg.

    The sensor is configured with one leg as the sensing bodies and the other leg as the filter
    partners, so the filtered force matrix contains only opposite-leg contacts. Contacts with the
    ground, torso, or arms are ignored by this term.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.normal_force_matrix_w_history.torch
    # forces: (num_envs, history, num_left_bodies, num_right_bodies, 3)
    return torch.any(torch.linalg.norm(forces, dim=-1) > threshold, dim=(1, 2, 3))
