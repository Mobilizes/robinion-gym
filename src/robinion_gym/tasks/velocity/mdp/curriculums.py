# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_velocity_range_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str,
    initial_range: dict[str, tuple[float, float]],
    final_range: dict[str, tuple[float, float]],
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly ramp an event's push velocity range between two bounds as training progresses.

    The range is interpolated with the environment step counter so the pushes of a fresh run start
    gentle and the full shove is only reached once the policy has had time to learn to keep its
    footing. The interpolated range is written in place into the event term's ``velocity_range``
    parameter, so the push term samples the ramped bounds for every push scheduled from then on.

    Args:
        env: The environment instance.
        env_ids: The environment indices (unused; present for the curriculum interface).
        event_name: Name of the event term whose ``velocity_range`` is ramped.
        initial_range: Per-axis velocity range (m/s) at ``start_step``.
        final_range: Per-axis velocity range (m/s) at ``end_step``. Axis names must match ``initial_range``.
        start_step: Environment step at which the ramp starts; before it the range stays at ``initial_range``.
        end_step: Environment step at which the ramp reaches ``final_range``.

    Returns:
        The ramp progress in ``[0, 1]`` and the current maximum absolute velocity bound, for logging.

    Raises:
        ValueError: If ``end_step`` is not greater than ``start_step`` or the axis names do not match.
    """
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step.")
    if initial_range.keys() != final_range.keys():
        raise ValueError("initial_range and final_range must have the same axes.")

    alpha = (env.common_step_counter - start_step) / (end_step - start_step)
    alpha = min(max(alpha, 0.0), 1.0)

    velocity_range = {
        axis: (
            low + alpha * (final_range[axis][0] - low),
            high + alpha * (final_range[axis][1] - high),
        )
        for axis, (low, high) in initial_range.items()
    }
    # the push term reads these bounds directly; write the new sample range in place
    event_cfg = env.event_manager.get_term_cfg(event_name)
    event_cfg.params["velocity_range"] = velocity_range
    env.event_manager.set_term_cfg(event_name, event_cfg)

    return {
        "progress": alpha,
        "velocity_max": max(abs(bound) for bounds in velocity_range.values() for bound in bounds),
    }
