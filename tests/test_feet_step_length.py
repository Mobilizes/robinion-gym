# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the inter-foot step-length reward."""

from types import SimpleNamespace

import torch

from robinion_gym.tasks.velocity.mdp.rewards import feet_step_length

_SENSOR_CFG = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
_ASSET_CFG = SimpleNamespace(name="robot", body_ids=[0, 1])


class _Scene:
    """Minimal scene exposing the contact sensor and the robot articulation."""

    def __init__(self, asset, sensor):
        self.sensors = {"contact_forces": sensor}
        self._asset = asset

    def __getitem__(self, name):
        return self._asset


def _make_env(
    foot_pos: torch.Tensor, first_contact: torch.Tensor, command: torch.Tensor
) -> SimpleNamespace:
    """Return a one-environment stand-in for the reward call."""
    num_envs = foot_pos.shape[0]
    root_quat = torch.zeros(num_envs, 4, dtype=foot_pos.dtype)
    root_quat[:, 0] = 1.0
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(
                torch=torch.cat([foot_pos, torch.zeros(num_envs, 2, 1)], dim=-1)
            ),
            root_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            root_quat_w=SimpleNamespace(torch=root_quat),
        )
    )
    sensor = SimpleNamespace(
        compute_first_contact=lambda dt: SimpleNamespace(torch=first_contact)
    )
    command_manager = SimpleNamespace(get_command=lambda name: command)
    return SimpleNamespace(
        scene=_Scene(asset, sensor),
        command_manager=command_manager,
        step_dt=0.005,
    )


def _reward(left_foot_x: float, right_foot_x: float, command_x: float) -> float:
    """Evaluate the reward with only the left foot touching down."""
    foot_pos = torch.tensor([[[left_foot_x, 0.0], [right_foot_x, 0.0]]])
    first_contact = torch.tensor([[True, False]])
    command = torch.tensor([[command_x, 0.0, 0.0]])
    env = _make_env(foot_pos, first_contact, command)
    return feet_step_length(env, "base_velocity", _SENSOR_CFG, _ASSET_CFG).item()


def test_landing_past_the_other_foot_is_rewarded_forward_and_backward():
    """Passing the planted foot must score the same in both travel directions."""
    forward = _reward(left_foot_x=0.04, right_foot_x=-0.04, command_x=0.4)
    backward = _reward(left_foot_x=-0.04, right_foot_x=0.04, command_x=-0.4)

    assert forward == backward
    assert forward > 0.99


def test_catch_up_step_is_penalized_forward_and_backward():
    """Landing level with the other foot is the leading-leg/catch-up gait, not a step."""
    forward = _reward(left_foot_x=0.04, right_foot_x=0.04, command_x=0.4)
    backward = _reward(left_foot_x=-0.04, right_foot_x=-0.04, command_x=-0.4)

    assert forward == backward
    assert forward < 0.2
