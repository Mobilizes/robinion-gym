# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the merged stand/walk task configuration."""

from robinion_gym.tasks.velocity.velocity_env_cfg import (
    RobinionVelocityEnvCfg,
    RobinionWalkInPlaceEnvCfg,
)


def test_merged_task_has_stand_walk_mode():
    """One run must expose the stand/walk bit to the commands, actor, critic, and rewards."""
    cfg = RobinionVelocityEnvCfg()

    assert hasattr(cfg.commands, "gait_mode")
    assert cfg.commands.base_velocity.gait_command_name == "gait_mode"
    assert "gait_mode" in cfg.observations.actor.__dict__
    assert "gait_mode" in cfg.observations.critic.__dict__
    for term_name in (
        "rew_stand_init_pose",
        "pen_stand_feet_air",
        "rew_feet_gait",
        "rew_feet_clearance",
        "rew_arm_swing",
    ):
        term = getattr(cfg.rewards, term_name)
        assert term.params["gait_command_name"] == "gait_mode"


def test_from_scratch_push_curriculum_is_slow():
    """From-scratch training needs a long unperturbed window and a slow push ramp."""
    cfg = RobinionVelocityEnvCfg()
    ramp = cfg.curriculum.push_velocity_ramp

    assert ramp.params["start_step"] >= 96000
    assert ramp.params["end_step"] >= 600000
    assert ramp.params["final_range"] == {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


def test_walk_in_place_is_zero_velocity_specialization():
    """The in-place task keeps the same mode bit but never translates."""
    cfg = RobinionWalkInPlaceEnvCfg()

    assert cfg.commands.base_velocity.gait_command_name == "gait_mode"
    assert "gait_mode" in cfg.observations.actor.__dict__
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.rewards.rew_stand_init_pose.params["gait_command_name"] == "gait_mode"
