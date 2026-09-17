# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generated task registrations."""

import gymnasium as gym

import robinion_gym.tasks  # noqa: F401


def test_task_registrations():
    """The generated tasks must expose valid environment and agent entry points."""
    expected = {
        "RobinionGym-Velocity-Robinion": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "robinion_gym.tasks.velocity.config.robinion.env_cfg:VelocityEnvCfg",
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
