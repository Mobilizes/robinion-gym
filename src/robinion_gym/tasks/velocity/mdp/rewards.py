# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
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


def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation from ``target_height`` with an L1 (absolute) kernel.

    Unlike :func:`isaaclab.envs.mdp.base_height_l2`, the gradient magnitude stays constant
    instead of shrinking with the error, so small height deviations are corrected as strongly
    as large ones, which keeps the base height tightly tracked.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w.torch[:, 2] - target_height)


def base_rp_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base roll and pitch deviation from level with an L1 (absolute) kernel.

    The base tilt is read from the x/y components of the projected gravity vector in the base
    frame, whose absolute values are summed. Unlike
    :func:`isaaclab.envs.mdp.flat_orientation_l2`, the gradient magnitude stays constant instead
    of shrinking with the error, so a small but persistent body tilt (for example, the residual
    lean left after recovering from a push) is corrected as strongly as a large one, driving the
    base back to level over time.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.projected_gravity_b.torch[:, :2]), dim=1)


class stand_still_drift(ManagerTermBase):
    """Reward the base holding its position while the commanded linear velocity is zero.

    An environment that is commanded to walk in place keeps stepping, so its base velocity
    oscillates with every step even when the stride has no net bias. Rewarding the instantaneous
    base velocity would fight that oscillation, so this term low-pass filters the base xy velocity
    (first-order exponential moving average) and applies an exponential kernel to the filtered
    value: only the slowly varying part of the motion, the drift that makes the robot wander, is
    measured, while the per-step oscillation is averaged out.

    The filter is restarted whenever the commanded linear velocity becomes (near) zero and on
    episode reset, so a previous walking state does not leak into the standing window. The term is
    zero for environments that are commanded to translate (``lin_vel_x``/``lin_vel_y`` above
    ``command_threshold``); turning-in-place commands (zero linear velocity) are anchored as well,
    since they should not translate either.

    Args:
        command_name: Name of the command term providing the velocity command.
        asset_cfg: The articulation whose base velocity is tracked.
        std: Standard deviation (m/s) of the exponential kernel; smaller values penalize drift harder.
        time_constant: Time constant (s) of the low-pass filter. Must be well above the gait period
            so the per-step oscillation is averaged out; around 1 s works for the 0.4 s gait clock.
        command_threshold: Linear command norm (m/s) below which the term is active.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # filtered base xy velocity (m/s) and standing mask, both per environment
        self._filtered_vel = torch.zeros(env.num_envs, 2, device=env.device)
        self._standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        std: float = 0.05,
        time_constant: float = 1.0,
        command_threshold: float = 1.0e-3,
    ) -> torch.Tensor:
        vel_xy = self._asset.data.root_lin_vel_b.torch[:, :2]
        cmd_xy = env.command_manager.get_command(command_name)[:, :2]
        standing = torch.norm(cmd_xy, dim=-1) <= command_threshold
        # restart the filter when the command becomes standing
        restart = standing & ~self._standing
        self._filtered_vel[restart] = 0.0
        # only update while standing; the mask also replaces the stored value elsewhere
        alpha = min(env.step_dt / time_constant, 1.0)
        self._filtered_vel = torch.where(
            standing.unsqueeze(-1),
            self._filtered_vel + alpha * (vel_xy - self._filtered_vel),
            self._filtered_vel,
        )
        self._standing = standing

        error = torch.sum(torch.square(self._filtered_vel), dim=-1)
        return standing.float() * torch.exp(-error / std**2)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._filtered_vel[env_ids] = 0.0
        self._standing[env_ids] = False


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


def feet_stance(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    swing_center: float = 0.25,
    swing_period: float = 0.4,
    force_threshold: float = 5.0,
    command_name=None,
) -> torch.Tensor:
    """Reward feet for being planted during their expected stance phase.

    For each foot, the reward is one while the foot is inside its support (stance) window and in
    contact with the ground. This complements :func:`feet_gait`, which rewards the opposite
    (swing) window, so a gait with a flight phase (both feet airborne) loses the support reward
    instead of getting the swing reward for free.

    If the commands are small (i.e. the agent is not supposed to walk), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        torch.norm(
            contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1
        )
        > force_threshold
    )

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(
        1
    )
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    # support is the complement of the swing window used by feet_gait
    in_swing_window = torch.abs(leg_phase - swing_center) < 0.5 * swing_period

    reward = torch.sum((~in_swing_window & is_contact).float(), dim=-1)

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
    min_height: float,
    std: float,
    tanh_mult: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing at least ``min_height`` off the ground.

    Only the shortfall below ``min_height`` is penalized, and only while the foot moves
    horizontally, so a swinging foot is free to lift higher than the minimum and a planted foot is
    left unpenalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    foot_z_shortfall = torch.square(torch.clamp(min_height - foot_z, min=0.0))
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_shortfall * foot_velocity_tanh
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


def feet_ahead_of_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize the leg assembly being placed in front of the base.

    The forward (base-frame x) offset of each foot from the base is computed in the base heading
    frame, ignoring roll and pitch. The penalty is the forward offset of the rearmost foot beyond
    ``margin``, so it is zero while the trailing foot is behind or under the base, which is the
    case for every normal alternating gait, and only grows when both feet are in front of the
    base (for example, a robot sitting back with its legs stretched forward). This does not limit
    the step length: a swing foot may reach as far forward as it likes as long as the other foot
    is still behind the body.

    Args:
        asset_cfg: The articulation and the foot bodies whose forward offset is measured.
        margin: Allowed forward offset (m) of the rearmost foot before the penalty starts.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]
    base_pos = asset.data.root_pos_w.torch[:, :2]
    base_yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)[2]

    # forward offset of each foot along the base heading (yaw only)
    delta = foot_pos - base_pos.unsqueeze(1)
    forward = (
        torch.cos(base_yaw).unsqueeze(1) * delta[..., 0]
        + torch.sin(base_yaw).unsqueeze(1) * delta[..., 1]
    )

    # the trailing foot is the one that must stay behind the body
    trailing = torch.min(forward, dim=-1).values
    return torch.clamp(trailing - margin, min=0.0)


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
