# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
from bam.actuator import TorchBackend
from bam.model import Model, load_model

from isaaclab.actuators import ActuatorBase, resolve_joint_parameter
from isaaclab.sim import SimulationContext
from isaaclab.utils.types import ArticulationActions

from .bam_actuator_cfg import BAMActuatorCfg


class BAMActuator(ActuatorBase):
    """Explicit actuator model backed by a BAM parameter file.

    Each simulation step, the current position target and joint state are passed
    through the BAM actuator's firmware control law
    (:meth:`bam.actuator.Actuator.compute_control`) and motor torque equation
    (:meth:`bam.actuator.Actuator.compute_torque`). The resulting effort is
    clipped by :attr:`actuator_effort_limit` before it is written to the
    articulation action.

    The BAM friction budget is not applied here: :meth:`ActuatorBase.compute`
    does not receive the external torque or link inertia that
    :meth:`bam.model.Model.compute_frictions` needs. Configure solver-level
    joint friction/damping separately, or extend this model if the
    load-dependent BAM friction is required.

    All joints in the group share one BAM model, so they must use the same servo
    type. Use one actuator group per BAM parameter file. Joints driven by several
    servos in parallel set :attr:`BAMActuatorCfg.num_servos`; their motor torque is
    multiplied accordingly.
    """

    cfg: BAMActuatorCfg
    """The configuration for the actuator model."""

    def __init__(
        self,
        cfg: BAMActuatorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        **kwargs,
    ):
        """Initialize the actuator.

        Args:
            cfg: The actuator configuration.
            joint_names: The names of the joints in the group.
            joint_ids: The indices of the joints in the articulation.
            num_envs: Number of environments.
            device: Device used for processing.
            stiffness: Unused. BAM computes the position control law internally.
            damping: Unused. BAM models motor damping through the back-EMF term.
            **kwargs: Additional arguments forwarded to :class:`~isaaclab.actuators.ActuatorBase`.
        """
        super().__init__(cfg, joint_names, joint_ids, num_envs, device, **kwargs)

        self.model: Model = load_model(self.cfg.model_path)
        # BAM must use its torch backend so the computation stays on the simulation device.
        self.model.actuator.backend = TorchBackend()
        if self.cfg.kp is not None:
            self.model.actuator.kp = self.cfg.kp
        if self.cfg.vin is not None:
            self.model.actuator.vin = self.cfg.vin

        self._dt = self.cfg.dt
        if self._dt is None:
            sim = SimulationContext.instance()
            self._dt = sim.get_physics_dt() if sim is not None else 1.0 / 60.0

        self._num_servos = resolve_joint_parameter(self.cfg.num_servos, 1.0, joint_names, num_envs, device)
        """Number of servos driving each joint in parallel, shape (num_envs, num_joints)."""

        self._control = torch.zeros_like(self.computed_effort)
        """Last control signal sent to the motor [V or A, depending on the BAM actuator]."""

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # ActuatorCollection passes None when every environment is reset.
        ids = slice(None) if env_ids is None else env_ids
        self.model.reset(ids)
        self._control[ids] = 0.0
        self.computed_effort[ids] = 0.0
        self.applied_effort[ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("BAMActuator requires joint position targets; no position command was provided.")

        control = self.model.actuator.compute_control(control_action.joint_positions, joint_pos, joint_vel, self._dt)
        if control is None:
            self._control.zero_()
        else:
            self._control.copy_(control)

        motor_effort = self.model.actuator.compute_torque(control, self.cfg.torque_enable, joint_pos, joint_vel)
        # Parallel servos see the same command and speed, so their torques add up.
        self.computed_effort = motor_effort * self._num_servos
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    """
    Properties.
    """

    @property
    def control(self) -> torch.Tensor:
        """Last control signal sent to the motor [V or A, depending on the BAM actuator]."""
        return self._control
