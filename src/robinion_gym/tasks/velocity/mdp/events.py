# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


def reset_non_finite_envs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset environments whose state has become non-finite (diverged).

    This is a safety guard against numerical blow-ups (NaN/inf in joint states or root pose)
    that can poison the RL rollout and crash training. It force-resets the affected
    environments through the standard reset pipeline, clears their already-computed reward,
    and flags them as terminated so the RL runner bootstraps correctly.

    Note:
        This is expected to be used as an ``interval`` event term with an interval of
        ``(0.0, 0.0)`` so that it runs every environment step.
    """
    asset = env.scene[asset_cfg.name]
    # check current state for non-finite values
    joint_pos = asset.data.joint_pos.torch[env_ids]
    joint_vel = asset.data.joint_vel.torch[env_ids]
    root_pos = asset.data.root_pos_w.torch[env_ids]
    root_quat = asset.data.root_quat_w.torch[env_ids]
    root_lin_vel = asset.data.root_lin_vel_w.torch[env_ids]
    root_ang_vel = asset.data.root_ang_vel_w.torch[env_ids]

    finite = (
        torch.isfinite(joint_pos).all(dim=-1)
        & torch.isfinite(joint_vel).all(dim=-1)
        & torch.isfinite(root_pos).all(dim=-1)
        & torch.isfinite(root_quat).all(dim=-1)
        & torch.isfinite(root_lin_vel).all(dim=-1)
        & torch.isfinite(root_ang_vel).all(dim=-1)
    )
    bad_ids = env_ids[~finite]
    if len(bad_ids) == 0:
        return

    # force a reset through the standard pipeline (refreshes sim + manager buffers)
    env._reset_idx(bad_ids)
    # clear the (possibly NaN) reward computed earlier this step
    env.reward_buf[bad_ids] = 0.0
    # inform the RL runner that these environments terminated this step
    env.reset_terminated[bad_ids] = True
    env.reset_time_outs[bad_ids] = False


def override_joint_pos_limits(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.5,
) -> None:
    """Override the joint position limits to a symmetric band around each joint's default position.

    The resulting limits are clamped to the original joint limits read from the asset so that this
    term can only tighten (never widen) the range of motion.

    Args:
        env: The environment instance.
        env_ids: The indices of the environments to apply the event to.
        asset_cfg: The asset configuration for the joints whose limits should be overridden.
        limit: The half-width (in rad) of the band around each joint's default position.
    """
    asset = env.scene[asset_cfg.name]
    # resolve indices
    if isinstance(env_ids, slice):
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        joint_ids = torch.as_tensor(joint_ids, device=asset.device)

    # default position of the concerned joints: (num_envs, num_joints)
    default_pos = asset.data.default_joint_pos.torch[env_ids[:, None], joint_ids]
    # original hard limits: (num_envs, num_joints, 2)
    orig_limits = asset.data.joint_pos_limits.torch[env_ids[:, None], joint_ids]
    # new limits = default position +/- limit, but never wider than the original limits
    new_low = torch.clamp(default_pos - limit, min=orig_limits[..., 0], max=orig_limits[..., 1])
    new_high = torch.clamp(default_pos + limit, min=orig_limits[..., 0], max=orig_limits[..., 1])
    new_limits = torch.stack([new_low, new_high], dim=-1)
    # write into the physics simulation
    asset.write_joint_position_limit_to_sim_index(limits=new_limits, joint_ids=joint_ids, env_ids=env_ids)


class push_robot_by_force(ManagerTermBase):
    """Push the robot with a short horizontal force, similar to a physical shove.

    Unlike :func:`push_by_setting_velocity`, which writes the sampled push directly into the
    root velocity, this term applies a horizontal force for ``duration_s`` so that the base
    velocity changes through the physics. Each body is driven with ``mass * dv / duration_s``,
    which makes the resulting base velocity change match ``velocity_range`` exactly.

    The term schedules its own pushes, so it must be used as an ``interval`` event with an
    interval of ``(0.0, 0.0)`` to be called on every environment step.

    Args:
        velocity_range: Desired base velocity change along the world x and y axes.
        duration_s: Duration of the shove in seconds.
        push_interval_range_s: Time between consecutive pushes, sampled per environment.
        asset_cfg: The articulation the pushes are applied to.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._force_w = torch.zeros(env.num_envs, self._asset.num_bodies, 3, device=env.device)
        self._active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._push_time_left = torch.zeros(env.num_envs, device=env.device)
        self._time_until_push = torch.zeros(env.num_envs, device=env.device)
        self._sample_time_until_push(slice(None))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        duration_s: float,
        push_interval_range_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        dt = env.step_dt

        self._push_time_left[self._active] -= dt
        expired = self._active & (self._push_time_left <= 0.0)
        if expired.any():
            expired_ids = expired.nonzero(as_tuple=False).flatten()
            self._force_w[expired_ids] = 0.0
            self._write(expired_ids)
            self._active[expired_ids] = False
            self._push_time_left[expired_ids] = 0.0

        self._time_until_push[~self._active] -= dt
        firing = (~self._active) & (self._time_until_push <= 0.0)
        if firing.any():
            firing_ids = firing.nonzero(as_tuple=False).flatten()
            bounds = torch.tensor(
                [velocity_range.get("x", (0.0, 0.0)), velocity_range.get("y", (0.0, 0.0))],
                device=env.device,
            )
            rand = torch.rand(len(firing_ids), 2, device=env.device)
            delta_v = bounds[:, 0] + rand * (bounds[:, 1] - bounds[:, 0])
            mass = self._asset.data.body_mass.torch[firing_ids]
            self._force_w[firing_ids] = 0.0
            self._force_w[firing_ids, :, :2] = mass.unsqueeze(-1) * (delta_v / duration_s).unsqueeze(1)
            self._active[firing_ids] = True
            self._push_time_left[firing_ids] = duration_s
            self._sample_time_until_push(firing_ids)
            self._write(firing_ids)

        active_ids = self._active.nonzero(as_tuple=False).flatten()
        if len(active_ids) > 0:
            self._write(active_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._active[env_ids] = False
        self._push_time_left[env_ids] = 0.0
        self._force_w[env_ids] = 0.0
        self._sample_time_until_push(env_ids)

    def _sample_time_until_push(self, env_ids: slice | torch.Tensor) -> None:
        lower, upper = self.cfg.params["push_interval_range_s"]
        num = self._env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._time_until_push[env_ids] = lower + torch.rand(num, device=self._env.device) * (upper - lower)

    def _write(self, env_ids: torch.Tensor) -> None:
        forces = self._force_w[env_ids]
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=None,
            env_ids=env_ids.to(dtype=torch.int32),
            is_global=True,
        )
