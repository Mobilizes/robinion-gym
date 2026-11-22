# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the merged stand/walk task configuration."""

from robinion_gym.tasks.velocity.velocity_env_cfg import (
    RobinionPushRecoveryEnvCfg,
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
        "rew_feet_stance",
        "rew_feet_step_length",
        "rew_feet_clearance",
        "rew_arm_swing",
    ):
        term = getattr(cfg.rewards, term_name)
        assert term.params["gait_command_name"] == "gait_mode"

    step_length = cfg.rewards.rew_feet_step_length
    assert step_length.params["command_name"] == "base_velocity"
    assert step_length.params["target_offset"] > 0.0
    assert step_length.params["std"] > 0.0


def test_from_scratch_push_curriculum_is_slow():
    """From-scratch training needs a long unperturbed window and a slow push ramp."""
    cfg = RobinionVelocityEnvCfg()
    ramp = cfg.curriculum.push_velocity_ramp

    assert ramp.params["start_step"] == 48000
    assert ramp.params["end_step"] == 300000
    assert ramp.params["final_range"] == {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}


def test_push_recovery_cfg_uses_full_commands_and_recovery_gating():
    """The push-recovery fine-tune must train at the play envelope and gate posture terms."""
    cfg = RobinionPushRecoveryEnvCfg()

    assert cfg.curriculum.push_velocity_ramp is None
    assert cfg.commands.base_velocity.ranges == cfg.commands.base_velocity.limit_ranges
    assert cfg.events.push_robot.params["velocity_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    }
    assert cfg.events.push_robot.params["recovery_s"] > 0.0
    for term_name in (
        "rew_stand_still",
        "rew_stand_init_pose",
        "pen_stand_feet_air",
        "pen_ankle_deviation",
        "pen_ankle_vel",
        "pen_joint_deviation_torso",
        "pen_arm_deviation",
        "pen_arm_wide",
        "pen_feet_slide",
        "pen_feet_distance",
    ):
        term = getattr(cfg.rewards, term_name)
        assert term.params["push_event_name"] == "push_robot"


def test_walk_in_place_is_zero_velocity_specialization():
    """The in-place task keeps the same mode bit but never translates."""
    cfg = RobinionWalkInPlaceEnvCfg()

    assert cfg.commands.base_velocity.gait_command_name == "gait_mode"
    assert "gait_mode" in cfg.observations.actor.__dict__
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.rewards.rew_stand_init_pose.params["gait_command_name"] == "gait_mode"
