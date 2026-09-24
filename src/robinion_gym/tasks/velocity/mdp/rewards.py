# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def joint_pos_target_l2(
    env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one (squared L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(deviation), dim=-1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    swing_center: float = 0.25,
    swing_period: float = 0.4,
    command_name=None,
) -> torch.Tensor:
    """Reward feet for swinging (being airborne) during their expected swing phase.

    For each foot, the reward is one if the foot is inside a narrow window around the swing
    center of its gait phase and the foot is not in contact with the ground. This is similar to
    a swing reward that rewards feet that are expected to swing but are currently airborne.

    If the commands are small (i.e. the agent is not supposed to walk), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0
    )

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(
        1
    )
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    # swing window around each leg phase center
    in_swing_window = torch.abs(leg_phase - swing_center) < 0.5 * swing_period

    reward = torch.sum((in_swing_window & ~is_contact).float(), dim=-1)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 1.0e-8
    return reward


def arm_swing_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    asset_cfg: SceneEntityCfg,
    amplitude: float,
    std: float,
    phase_offset: float = 0.0,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward arm swinging that is phase-locked to the same-side leg gait.

    Each arm's shoulder joint is driven to track a sinusoidal reference with the same ``period``
    and per-arm ``offset`` as the leg gait, so the left arm swings with the left leg and the right
    arm with the right leg. Setting ``phase_offset`` to :math:`\\pi` makes each arm swing while the
    same-side leg is in its support (stance) phase, counter-balancing the legs and improving
    stability.

    The oscillation is centered on the neutral (zero) joint position, so the arms swing about the
    hanging pose rather than a fixed forward offset. If ``command_name`` is given, the reward is
    zero whenever the commanded velocity is (near) zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(
        1
    )
    phases = [(global_phase + off) % 1.0 for off in offset]
    arm_phase = torch.cat(phases, dim=-1)  # (num_envs, num_arms)

    target = amplitude * torch.sin(2.0 * torch.pi * arm_phase + phase_offset)
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(joint_pos - target), dim=-1)
    reward = torch.exp(-error / std**2)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 1.0e-8
    return reward


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    The reward is high when each foot is near ``target_height`` while moving horizontally, so it
    only shapes the swing (moving) foot and leaves planted feet unpenalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 1.0e-8
    return reward


def feet_flat_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    swing_center: float = 0.25,
    swing_period: float = 0.4,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Reward feet for being flat and fully planted on the ground during their stance phase.

    For each foot, the reward is the squared cosine of the angle between the foot's local up axis
    and the world up axis (i.e. how flat the foot is), but only while the foot is in its support
    (stance) phase of the gait and in contact with the ground. This rewards the robot for
    planting the whole foot on the ground instead of touching with only part of the sole
    (e.g. the toe or heel), and does not penalize the foot being airborne during its swing phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        torch.norm(
            contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1
        )
        > force_threshold
    )

    # gait phase per foot
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(
        1
    )
    phases = []
    for offset_ in offset:
        phases.append((global_phase + offset_) % 1.0)
    leg_phase = torch.cat(phases, dim=-1)

    # swing window around each leg phase center; support is the complement
    in_swing_window = torch.abs(leg_phase - swing_center) < 0.5 * swing_period
    in_support = ~in_swing_window

    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=foot_quat.dtype)
    up_world = quat_apply(foot_quat, up_local.expand_as(foot_quat[..., :3]))

    flatness = torch.clamp(up_world[..., 2], min=0.0)
    return torch.sum(flatness.pow(2) * is_contact.float() * in_support.float(), dim=-1)


def feet_flat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward feet for staying parallel to the ground over the whole gait cycle.

    For each foot, the reward is the squared cosine of the angle between the foot's local up axis
    and the world up axis (i.e. how flat the foot is, independent of its heading). Unlike
    :func:`feet_flat_contact`, this is applied at all times rather than only during stance, so the
    swing foot is discouraged from tilting (toe-up/down or rolling sideways) while it is airborne,
    which encourages a flat, controlled foot pose at touchdown. Yaw (heading) is intentionally not
    constrained here; see :func:`feet_yaw_diff` and :func:`feet_yaw_mean`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=foot_quat.dtype)
    up_world = quat_apply(foot_quat, up_local.expand_as(foot_quat[..., :3]))

    flatness = torch.clamp(up_world[..., 2], min=0.0)
    return torch.sum(flatness.pow(2), dim=-1)


def _get_feet_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Yaw angle (world z) of each foot body. Shape is (N, num_feet)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # (N, num_feet, 4)
    num_feet = quat.shape[1]
    yaw = euler_xyz_from_quat(quat.reshape(-1, 4))[2]
    return yaw.reshape(-1, num_feet)


def feet_yaw_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the yaw (heading) difference between the two feet.

    The reward is the squared angular difference between the feet wrapped to (-pi, pi).
    """
    feet_yaw = _get_feet_yaw(env, asset_cfg)
    return torch.square(
        (feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi
    )


def feet_yaw_mean(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the yaw (heading) difference between the base and the mean foot yaw.

    The reward is the squared angular difference between the base yaw and the mean foot
    yaw, wrapped to (-pi, pi).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_yaw = _get_feet_yaw(env, asset_cfg)
    feet_yaw_mean = feet_yaw.mean(dim=-1) + torch.pi * (
        torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi
    )
    base_yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)[2]
    return torch.square(
        (base_yaw - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi
    )


def feet_distance(
    env: ManagerBasedRLEnv,
    left_foot_cfg: SceneEntityCfg,
    right_foot_cfg: SceneEntityCfg,
    target_distance: float = 0.25,
    command_name: str | None = None,
    full_gate_vel: float = 0.3,
) -> torch.Tensor:
    """Penalize the left foot deviating from the target separation to the right foot.

    The signed lateral (base-frame y) separation ``left - right`` is compared against
    ``target_distance`` and the absolute deviation is penalized, so both a narrow/crossed stance
    and an overly wide stance are discouraged.

    If ``command_name`` is given, the term's weight is gated by the commanded lateral velocity:
    the penalty runs at full weight when the command is straight (``cmd_y == 0``) and fades to
    zero as ``|cmd_y|`` reaches ``full_gate_vel``, since side-stepping naturally changes the
    stance and the planted-foot separation constraint is less meaningful then.
    """
    asset: Articulation = env.scene[left_foot_cfg.name]

    base_yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)[2]
    left_pos = asset.data.body_pos_w.torch[:, left_foot_cfg.body_ids[0]]
    right_pos = asset.data.body_pos_w.torch[:, right_foot_cfg.body_ids[0]]

    lateral = torch.cos(base_yaw) * (left_pos[:, 1] - right_pos[:, 1]) - torch.sin(
        base_yaw
    ) * (left_pos[:, 0] - right_pos[:, 0])

    value = torch.clip(torch.abs(target_distance - lateral), max=0.5)
    if command_name is not None:
        cmd_y = env.command_manager.get_command(command_name)[:, 1]
        gate = torch.clip(1.0 - torch.abs(cmd_y) / full_gate_vel, min=0.0, max=1.0)
        value = value * gate
    return value
