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
