# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .bam_actuator import BAMActuator

_BAM_MODEL_RELATIVE_PATH = Path("move_actuator/bam-xseries/params/xh540/m6.json")
"""Default BAM parameter file, relative to the workspace root."""


def _default_model_path() -> str:
    """Return the absolute path of the default BAM model, if it can be found."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _BAM_MODEL_RELATIVE_PATH
        if candidate.is_file():
            return str(candidate)
    # Fall back to the path used by the project's run commands (repository root as CWD).
    return "../move_actuator/bam-xseries/params/xh540/m6.json"


@configclass
class BAMActuatorCfg(ActuatorBaseCfg):
    """Configuration for an explicit actuator driven by a BAM model.

    The BAM model supplies both the firmware position-control law
    (:meth:`bam.actuator.Actuator.compute_control`) and the motor torque equation
    (:meth:`bam.actuator.Actuator.compute_torque`). All joints in the group share
    one model, so they must use the same servo type.
    """

    class_type: type[BAMActuator] | str = "{DIR}.bam_actuator:BAMActuator"

    model_path: str = _default_model_path()
    """Path to the BAM parameter JSON file.

    Defaults to the identified XH540-W270 ``m6`` model in the sibling
    ``move_actuator/bam-xseries`` checkout.
    """

    kp: float | None = None
    """Firmware position gain. If None, the value in the BAM actuator is used."""

    vin: float | None = None
    """Supply voltage [V]. If None, the value in the BAM actuator is used."""

    torque_enable: bool = True
    """Whether the motor torque is enabled. When False, the motor torque is zero."""

    dt: float | None = None
    """Time step [s] passed to the BAM control law.

    If None, the simulation physics time step is used.
    """

    stiffness: dict[str, float] | float | None = 0.0
    """Unused. BAM computes the position control law internally."""

    damping: dict[str, float] | float | None = 0.0
    """Unused. BAM models motor damping through the back-EMF term."""
