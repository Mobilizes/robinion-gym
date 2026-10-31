# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import ModelFlags

from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import read_group_parameter, write_group_parameter
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


def reset_non_finite_envs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | None = None,
    max_joint_vel: float = 50.0,
    max_root_lin_vel: float = 20.0,
    max_root_ang_vel: float = 50.0,
    max_action: float = 10.0,
) -> None:
    """Reset environments whose state has diverged (non-finite, implausible, or action runaway).

    This is a safety guard against numerical blow-ups that can poison the RL rollout and crash
    training. It force-resets the affected environments through the standard reset pipeline,
    clears their already-computed reward, and flags them as terminated so the RL runner
    bootstraps correctly.

    A diverging rigid body does not necessarily become non-finite immediately: it can stay
    finite for several steps while producing enormous observations and rewards. Those values
    are far more damaging to the value function than one bad step, so states beyond the
    ``max_*`` thresholds are reset as well. The action check is a backstop for a policy whose
    outputs run away faster than the robot state diverges, for example through the
    ``last_action`` observation; keep ``max_action`` above any action clip range in use so it
    only fires on genuine anomalies.

    Note:
        This is expected to be used as an ``interval`` event term with an interval of
        ``(0.0, 0.0)`` so that it runs every environment step.

    Args:
        env: The environment instance.
        env_ids: The indices of the environments to apply the event to.
        asset_cfg: The asset configuration for the articulation to check.
        max_joint_vel: Maximum plausible absolute joint velocity [rad/s]. Defaults to ``50.0``.
        max_root_lin_vel: Maximum plausible root linear velocity [m/s]. Defaults to ``20.0``.
        max_root_ang_vel: Maximum plausible root angular velocity [rad/s]. Defaults to ``50.0``.
        max_action: Maximum plausible absolute action value. Defaults to ``10.0``.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    # check current state for non-finite or implausible values
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
    plausible = (
        (joint_vel.abs() <= max_joint_vel).all(dim=-1)
        & (torch.linalg.norm(root_lin_vel, dim=-1) <= max_root_lin_vel)
        & (torch.linalg.norm(root_ang_vel, dim=-1) <= max_root_ang_vel)
        & (env.action_manager.action[env_ids].abs().amax(dim=-1) <= max_action)
    )
    bad_ids = env_ids[~(finite & plausible)]
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
    asset_cfg: SceneEntityCfg | None = None,
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
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
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


class randomize_foot_contact(ManagerTermBase):
    """Randomize the foot contact model per environment on Newton's MJWarp solver.

    Sliding friction is the maximum of the two contacting shapes' coefficients in MuJoCo, so the
    ground plane is authored at the lower bound and the feet are sampled over the full effective
    range. Torsional (spin) and rolling friction are sampled on the feet as well; the ground's much
    smaller defaults leave the sampled foot values in control. The MuJoCo contact reference is
    randomized through the foot contact stiffness and damping: ``(time_constant, damping_ratio)``
    is sampled and converted with ``kd = 2 / time_constant`` and
    ``ke = (kd / (2 * damping_ratio)) ** 2``. The effective solref is the solmix-weighted average of
    the foot and ground values, so the sampled foot range is wider than the effective range.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. ``None`` or a full slice randomizes every environment.
        asset_cfg: The articulation and foot bodies whose collision shapes are randomized.
        friction_range: Effective sliding-friction range.
        torsional_friction_range: Torsional (spin) friction range.
        rolling_friction_range: Rolling friction range.
        solref_time_constant_range: Foot contact time-constant range [s].
        solref_damping_ratio_range: Foot contact damping-ratio range [-].
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.asset_cfg = asset_cfg
        self._newton_manager = env.sim.backend_manager
        model = self._newton_manager.backend.model
        if model is None:
            raise RuntimeError("randomize_foot_contact requires Newton's MJWarp backend")

        root_view = self.asset._root_view
        self._mu = wp.to_torch(root_view.get_attribute("shape_material_mu", model)[:, 0])
        self._mu_torsional = wp.to_torch(root_view.get_attribute("shape_material_mu_torsional", model)[:, 0])
        self._mu_rolling = wp.to_torch(root_view.get_attribute("shape_material_mu_rolling", model)[:, 0])
        self._ke = wp.to_torch(root_view.get_attribute("shape_material_ke", model)[:, 0])
        self._kd = wp.to_torch(root_view.get_attribute("shape_material_kd", model)[:, 0])

        if asset_cfg.body_ids != slice(None):
            # The view's shape axis is not grouped by body, so select each backend body's own
            # shape indices (the same mapping the standard material event uses).
            body_shapes = root_view.body_shapes
            backend_body_ids = self.asset.map_body_ids_to_backend(asset_cfg.body_ids)
            shape_indices = [shape_id for body_id in backend_body_ids for shape_id in body_shapes[body_id]]
            self._shape_indices = torch.tensor(shape_indices, dtype=torch.long)
        else:
            self._shape_indices = torch.arange(self._mu.shape[1], dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        friction_range: tuple[float, float],
        torsional_friction_range: tuple[float, float],
        rolling_friction_range: tuple[float, float],
        solref_time_constant_range: tuple[float, float],
        solref_damping_ratio_range: tuple[float, float],
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return

        device = env.device
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
        shape_idx = self._shape_indices.to(device)
        num_shapes = len(shape_idx)

        def sample(value_range: tuple[float, float]) -> torch.Tensor:
            return torch.empty((num_envs, num_shapes), device=device).uniform_(*value_range)

        self._mu[env_rows, shape_idx] = sample(friction_range)
        self._mu_torsional[env_rows, shape_idx] = sample(torsional_friction_range)
        self._mu_rolling[env_rows, shape_idx] = sample(rolling_friction_range)

        time_constant = sample(solref_time_constant_range)
        damping_ratio = sample(solref_damping_ratio_range)
        stiffness_damping = 2.0 / time_constant
        self._kd[env_rows, shape_idx] = stiffness_damping
        self._ke[env_rows, shape_idx] = torch.square(stiffness_damping / (2.0 * damping_ratio))

        self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


class randomize_bam_actuator(ManagerTermBase):
    """Randomize BAM servo strength, firmware gain, and gearbox friction per environment.

    The nominal values are captured once at term construction, so repeated resets sample around the
    identified servo model instead of accumulating previous randomizations. Command latency and
    supply voltage are configured on :class:`~isaaclab.actuators.BamActuatorCfg` (they are shared
    drive settings), while the per-DOF ``kp_scale``, ``max_effort``, and ``friction_scale`` arrays
    are written here.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: The articulation whose BAM actuator groups are randomized.
        kp_scale_range: Firmware position-gain multiplier range.
        max_effort_scale_range: Torque-limit multiplier range.
        friction_scale_range: Gearbox friction-budget multiplier range. ``None`` leaves it unchanged.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self._groups: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for name, actuator_cfg in self.asset.cfg.actuators.items():
            if not isinstance(actuator_cfg, BamActuatorCfg):
                continue
            self._groups.append(
                (
                    name,
                    read_group_parameter(self.asset.actuators, name, "drive", "kp_scale").clone(),
                    read_group_parameter(self.asset.actuators, name, "drive", "max_effort").clone(),
                    read_group_parameter(self.asset.actuators, name, "drive", "friction_scale").clone(),
                )
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        kp_scale_range: tuple[float, float],
        max_effort_scale_range: tuple[float, float],
        friction_scale_range: tuple[float, float] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return

        device = env.device
        for name, kp_nominal, effort_nominal, friction_nominal in self._groups:
            num_joints = kp_nominal.shape[1]

            def scale(value_range: tuple[float, float], columns: int) -> torch.Tensor:
                return torch.empty((num_envs, columns), device=device).uniform_(*value_range)

            write_group_parameter(
                self.asset.actuators,
                name,
                "drive",
                "kp_scale",
                values=kp_nominal[env_ids] * scale(kp_scale_range, num_joints),
                env_ids=env_ids,
            )
            write_group_parameter(
                self.asset.actuators,
                name,
                "drive",
                "max_effort",
                values=effort_nominal[env_ids] * scale(max_effort_scale_range, num_joints),
                env_ids=env_ids,
            )
            if friction_scale_range is not None:
                write_group_parameter(
                    self.asset.actuators,
                    name,
                    "drive",
                    "friction_scale",
                    values=friction_nominal[env_ids] * scale(friction_scale_range, num_joints),
                    env_ids=env_ids,
                )


class push_robot_by_force(ManagerTermBase):
    """Push the robot with a short horizontal force, similar to a physical shove.

    Unlike :func:`push_by_setting_velocity`, which writes the sampled push directly into the
    root velocity, this term applies a horizontal force for ``duration_s`` so that the base
    velocity changes through the physics. Each body is driven with ``mass * dv / duration_s``,
    which makes the resulting base velocity change match ``velocity_range`` exactly.

    The term schedules its own pushes, so it must be used as an ``interval`` event with an
    interval of ``(0.0, 0.0)`` to be called on every environment step.

    When the ``debug_vis`` parameter is enabled, a red arrow is drawn above the robot base for
    every push. The arrow points along the net push force and its length scales with the force
    magnitude. It stays visible after the push ends, showing the most recent push until the next
    push replaces it (and is cleared when the environment resets).

    Args:
        velocity_range: Desired base velocity change along the world x and y axes.
        duration_s: Duration of the shove in seconds.
        push_interval_range_s: Time between consecutive pushes, sampled per environment.
        asset_cfg: The articulation the pushes are applied to.
        debug_vis: Whether to draw the red force arrow. Defaults to False.
        marker_pos_offset: Offset [m] from the robot root where the force arrow is drawn.
            Defaults to ``(0.0, 0.0, 0.5)``.
        force_arrow_scale: Arrow length in meters per newton of net push force. Defaults to ``0.01``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._force_w = torch.zeros(env.num_envs, self._asset.num_bodies, 3, device=env.device)
        self._active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._push_time_left = torch.zeros(env.num_envs, device=env.device)
        self._time_until_push = torch.zeros(env.num_envs, device=env.device)
        self._sample_time_until_push(slice(None))
        # most recent push force per environment, kept for the persistent debug arrow
        self._last_force_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._show_arrow = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # red arrow marker that shows the push force (created only with debug visualization)
        self._force_visualizer: VisualizationMarkers | None = None
        if cfg.params.get("debug_vis", False):
            self._force_visualizer = VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Events/push_force")
            )
            self._force_visualizer.set_visibility(False)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        duration_s: float,
        push_interval_range_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
        debug_vis: bool = False,
        marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.5),
        force_arrow_scale: float = 0.01,
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
            # remember the new push for the persistent debug arrow
            self._last_force_w[firing_ids] = self._force_w[firing_ids].sum(dim=1)
            self._show_arrow[firing_ids] = True

        active_ids = self._active.nonzero(as_tuple=False).flatten()
        if len(active_ids) > 0:
            self._write(active_ids)

        if debug_vis and self._force_visualizer is not None:
            self._update_force_visualization(marker_pos_offset, force_arrow_scale)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._active[env_ids] = False
        self._push_time_left[env_ids] = 0.0
        self._force_w[env_ids] = 0.0
        self._last_force_w[env_ids] = 0.0
        self._show_arrow[env_ids] = False
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

    def _update_force_visualization(
        self,
        marker_pos_offset: tuple[float, float, float],
        force_arrow_scale: float,
    ) -> None:
        """Draw a red arrow at the robot base for the most recent push of every environment.

        The arrow stays visible after a push ends so that the most recent push force keeps being
        displayed until the next push replaces it.
        """
        visualizer = self._force_visualizer
        if visualizer is None:
            return
        # hide the arrows until an environment has been pushed for the first time
        shown_ids = self._show_arrow.nonzero(as_tuple=False).flatten()
        if len(shown_ids) == 0:
            visualizer.set_visibility(False)
            return

        # net force (N) of the most recent push, summed over the bodies
        force_w = self._last_force_w[shown_ids]
        # place the arrow above the robot base
        offset = torch.tensor(
            marker_pos_offset,
            device=self.device,
            dtype=force_w.dtype,
        )
        translations = self._asset.data.root_pos_w.torch[shown_ids].clone() + offset
        # point the arrow along the horizontal force direction (world frame)
        heading = torch.atan2(force_w[:, 1], force_w[:, 0])
        orientations = quat_from_euler_xyz(torch.zeros_like(heading), torch.zeros_like(heading), heading)
        # scale the arrow length with the force magnitude
        default_scale = torch.tensor(visualizer.cfg.markers["arrow"].scale, device=self.device, dtype=force_w.dtype)
        scales = default_scale.repeat(len(shown_ids), 1)
        scales[:, 0] *= torch.linalg.norm(force_w[:, :2], dim=1) * force_arrow_scale

        visualizer.set_visibility(True)
        visualizer.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            environment_ids=shown_ids,
        )
