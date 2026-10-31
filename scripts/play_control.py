# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Robinion policy with live keyboard or gamepad velocity commands.

The script mirrors the ``isaaclab play`` workflow for the registered
``RobinionGym-Velocity-Robinion`` task, but forces ``num_envs=1`` and replaces the random
velocity command generator with a teleoperation command built from live input:

* Keyboard: ``W``/``S`` forward/back, ``A``/``D`` turn, ``Q``/``E`` strafe,
  ``F`` shove the robot, ``R`` reset the robot.
* Gamepad: left stick forward/strafe with analog strength, right stick turn with analog
  strength, ``A`` shove the robot, ``X`` reset the robot.

Keyboard and gamepad can be used at the same time. Command changes are low-pass filtered so
taps and stick motion do not step the policy, and the gamepad sticks use a radial deadzone.

The manual shove applies a horizontal force to the robot for a short duration, with the
direction fixed relative to the robot heading when the key/button is pressed. A red arrow
is drawn above the robot while the force is active.

.. code-block:: bash

    # Latest checkpoint, keyboard/gamepad control, Kit window (input needs the Kit window)
    ./isaaclab.sh -p scripts/play_control.py --checkpoint latest

    # Control only the forward/back axis and shove a little harder
    ./isaaclab.sh -p scripts/play_control.py --checkpoint latest \
        --max_lin_vel_y 0.0 --max_ang_vel_z 0.0 --push_force 80

    # Use a specific checkpoint and a custom task
    ./isaaclab.sh -p scripts/play_control.py \
        --checkpoint logs/rsl_rl/robinion_velocity/walk-v5/model_9999.pt \
        --task RobinionGym-Velocity-Robinion
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

import robinion_gym.tasks  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class InputState:
    """One frame of normalized teleoperation input."""

    lin_x: float = 0.0
    lin_y: float = 0.0
    ang_z: float = 0.0
    push: bool = False
    reset: bool = False


class KeyboardGamepadInput:
    """Poll the Isaac Sim keyboard and gamepad for SE(2) commands and edge-triggered actions.

    The devices are polled instead of driven by callbacks so a lost focus cannot leave a key
    stuck down. Keyboard and gamepad contributions are summed and clamped, which allows using
    both devices at the same time.
    """

    _KEYBOARD_BINDINGS = {
        "forward": "W",
        "backward": "S",
        "turn_left": "A",
        "turn_right": "D",
        "strafe_left": "Q",
        "strafe_right": "E",
        "push": "F",
        "reset": "R",
    }

    def __init__(self, mode: str, deadzone: float, keyboard_gain: float):
        """Create the input source.

        Args:
            mode: ``"keyboard"``, ``"gamepad"``, or ``"both"``.
            deadzone: Radial deadzone applied to the gamepad sticks.
            keyboard_gain: Scale factor applied to keyboard axes.
        """
        import carb.input
        import omni.appwindow

        self._carb = carb.input
        self._input = carb.input.acquire_input_interface()
        self._window = omni.appwindow.get_default_app_window()
        if self._window is None:
            raise RuntimeError(
                "No Isaac Sim application window is available. Keyboard and gamepad input need "
                "the Kit window; run with '--viz kit' and without '--headless'."
            )

        self._mode = mode
        self._deadzone = max(0.0, min(deadzone, 0.95))
        self._keyboard_gain = keyboard_gain
        self._keyboard = self._window.get_keyboard() if mode in ("keyboard", "both") else None
        if mode in ("keyboard", "both") and self._keyboard is None:
            raise RuntimeError("No keyboard device was found in the Isaac Sim application window.")

        self._gamepad = None
        self._gamepad_name = ""
        if mode in ("gamepad", "both"):
            self._refresh_gamepad()
            if self._gamepad is None:
                if mode == "gamepad":
                    raise RuntimeError(
                        "No gamepad was found. Connect a controller before starting the script, "
                        "or use '--input keyboard'."
                    )
                print(
                    "[WARN] No gamepad detected; keyboard control is still active. "
                    "Connect a controller and it will be picked up automatically.",
                    flush=True,
                )

        self._prev_push = False
        self._prev_reset = False

    def _refresh_gamepad(self) -> None:
        """Look up the first connected gamepad."""
        try:
            gamepad = self._window.get_gamepad(0)
        except Exception:
            gamepad = None
        if gamepad is None:
            return
        try:
            name = self._input.get_gamepad_name(gamepad)
        except Exception:
            return
        if name:
            self._gamepad = gamepad
            changed = name != self._gamepad_name
            self._gamepad_name = name
            if changed:
                print(f"[INFO] Gamepad connected: {name}", flush=True)

    def _key_down(self, name: str) -> bool:
        """Return whether a bound keyboard key is held."""
        key = getattr(self._carb.KeyboardInput, self._KEYBOARD_BINDINGS[name])
        return bool(self._input.get_keyboard_button_flags(self._keyboard, key) & self._carb.BUTTON_FLAG_DOWN)

    def _gamepad_down(self, name: str) -> bool:
        """Return whether a gamepad button is held."""
        if self._gamepad is None:
            return False
        button = getattr(self._carb.GamepadInput, name)
        try:
            flags = self._input.get_gamepad_button_flags(self._gamepad, button)
        except Exception:
            self._gamepad = None
            return False
        return bool(flags & self._carb.BUTTON_FLAG_DOWN)

    def _axis(self, positive: str, negative: str) -> float:
        """Return ``positive - negative`` for one analog gamepad axis."""
        if self._gamepad is None:
            return 0.0
        try:
            value = self._input.get_gamepad_value(
                self._gamepad, getattr(self._carb.GamepadInput, positive)
            ) - self._input.get_gamepad_value(self._gamepad, getattr(self._carb.GamepadInput, negative))
        except Exception:
            self._gamepad = None
            return 0.0
        return float(value)

    def _gamepad_sticks(self) -> tuple[float, float, float]:
        """Return the gamepad stick axes after applying a radial deadzone to the left stick."""
        forward = self._axis("LEFT_STICK_UP", "LEFT_STICK_DOWN")
        right = self._axis("LEFT_STICK_RIGHT", "LEFT_STICK_LEFT")
        # Player view: up is forward (+x) and right is -y in the robot base frame.
        lin_x = forward
        lin_y = -right
        magnitude = math.hypot(lin_x, lin_y)
        if magnitude <= self._deadzone:
            lin_x = 0.0
            lin_y = 0.0
        else:
            scale = (magnitude - self._deadzone) / (1.0 - self._deadzone)
            lin_x *= scale / magnitude
            lin_y *= scale / magnitude

        turn_right = self._axis("RIGHT_STICK_RIGHT", "RIGHT_STICK_LEFT")
        if abs(turn_right) <= self._deadzone:
            turn_right = 0.0
        else:
            turn_right = math.copysign((abs(turn_right) - self._deadzone) / (1.0 - self._deadzone), turn_right)
        # Right stick right turns clockwise, which is negative yaw in the base frame.
        return lin_x, lin_y, -turn_right

    def _keyboard_sticks(self) -> tuple[float, float, float]:
        """Return the keyboard axes as ``-1``/``0``/``1`` values."""
        if self._keyboard is None:
            return 0.0, 0.0, 0.0
        forward = float(self._key_down("forward")) - float(self._key_down("backward"))
        strafe = float(self._key_down("strafe_left")) - float(self._key_down("strafe_right"))
        yaw = float(self._key_down("turn_left")) - float(self._key_down("turn_right"))
        return (
            self._keyboard_gain * forward,
            self._keyboard_gain * strafe,
            self._keyboard_gain * yaw,
        )

    def advance(self) -> InputState:
        """Poll the devices and return one normalized command frame."""
        if self._gamepad is None and self._mode in ("gamepad", "both"):
            self._refresh_gamepad()

        lin_x_kb, lin_y_kb, ang_z_kb = self._keyboard_sticks()
        lin_x_gp, lin_y_gp, ang_z_gp = self._gamepad_sticks()

        push_down = (self._keyboard is not None and self._key_down("push")) or self._gamepad_down("A")
        reset_down = (self._keyboard is not None and self._key_down("reset")) or self._gamepad_down("X")
        push = push_down and not self._prev_push
        reset = reset_down and not self._prev_reset
        self._prev_push = push_down
        self._prev_reset = reset_down

        return InputState(
            lin_x=max(-1.0, min(1.0, lin_x_kb + lin_x_gp)),
            lin_y=max(-1.0, min(1.0, lin_y_kb + lin_y_gp)),
            ang_z=max(-1.0, min(1.0, ang_z_kb + ang_z_gp)),
            push=push,
            reset=reset,
        )


class ForcePusher:
    """Apply a short horizontal force to the robot and draw a debug arrow for it."""

    def __init__(
        self,
        asset: Articulation,
        force: float | None,
        duration_s: float,
        cooldown_s: float,
        direction: str,
        body_expr: str | None,
        visualize: bool,
        arrow_scale: float,
    ):
        """Create the pusher.

        Args:
            asset: The robot articulation.
            force: Net horizontal force in newtons. ``None`` uses ``2.5 * total_mass``, which
                changes the base velocity by about 0.5 m/s over the default 0.2 s duration.
            duration_s: Time the force stays applied.
            cooldown_s: Time before another push can be triggered.
            direction: Push direction relative to the robot heading.
            body_expr: Optional body-name regex. ``None`` applies the force to every body.
            visualize: Whether to draw the force arrow.
            arrow_scale: Arrow length in meters per newton.
        """
        self._asset = asset
        self._env_ids = torch.tensor([0], dtype=torch.int32, device=asset.device)
        self._duration_s = duration_s
        self._cooldown_s = cooldown_s
        self._direction = direction
        self._arrow_scale = arrow_scale
        self._remaining_s = 0.0
        self._cooldown_left_s = 0.0
        self._force_dir_w = torch.zeros(3, device=asset.device)
        self._net_force = 0.0

        self._body_ids = None
        if body_expr is not None:
            body_ids, body_names = asset.find_bodies(body_expr, preserve_order=True)
            if len(body_ids) == 0:
                print(
                    f"[WARN] Push body regex '{body_expr}' matched no bodies; pushing every body instead.",
                    flush=True,
                )
            else:
                self._body_ids = torch.tensor(body_ids, dtype=torch.int32, device=asset.device)
                print(f"[INFO] Push bodies: {body_names}", flush=True)

        mass = asset.data.body_mass.torch[0]
        if self._body_ids is not None:
            mass = mass[self._body_ids]
        self._total_mass = float(mass.sum())
        self._net_force = float(force) if force is not None else 2.5 * self._total_mass
        if self._total_mass <= 0.0:
            raise RuntimeError("The robot has no body mass, so a force push cannot be scaled.")

        print(
            f"[INFO] Push force: {self._net_force:.1f} N for {duration_s:.2f} s "
            f"(robot mass {self._total_mass:.2f} kg, direction {direction})",
            flush=True,
        )

        self._visualizer: VisualizationMarkers | None = None
        if visualize:
            try:
                self._visualizer = VisualizationMarkers(
                    RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/PlayControl/push_force")
                )
                self._visualizer.set_visibility(False)
            except Exception as exc:  # markers need a stage-capable visualizer
                print(f"[WARN] Push visualization disabled: {exc}", flush=True)

    @property
    def net_force(self) -> float:
        """Net horizontal force of a push in newtons."""
        return self._net_force

    @property
    def duration_s(self) -> float:
        """Duration of a push in seconds."""
        return self._duration_s

    def _write_force(self, force: torch.Tensor) -> None:
        """Write a horizontal force to the selected bodies."""
        torques = torch.zeros_like(force)
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            torques=torques,
            body_ids=self._body_ids,
            env_ids=self._env_ids,
            is_global=True,
        )

    def trigger(self) -> bool:
        """Start a push in the configured direction if no push is active."""
        if self._remaining_s > 0.0 or self._cooldown_left_s > 0.0:
            return False

        root_quat = self._asset.data.root_quat_w.torch[0].unsqueeze(0)
        heading = float(euler_xyz_from_quat(root_quat)[2])
        if self._direction == "random":
            angle = random.uniform(-math.pi, math.pi)
        else:
            angle = (
                heading
                + {
                    "forward": 0.0,
                    "left": 0.5 * math.pi,
                    "backward": math.pi,
                    "right": -0.5 * math.pi,
                }[self._direction]
            )
        self._force_dir_w = torch.tensor([math.cos(angle), math.sin(angle), 0.0], device=self._asset.device)

        mass = self._asset.data.body_mass.torch[0]
        if self._body_ids is not None:
            mass = mass[self._body_ids]
        weights = mass / mass.sum()
        forces = self._net_force * weights.unsqueeze(-1) * self._force_dir_w.unsqueeze(0)
        self._write_force(forces.unsqueeze(0))

        self._remaining_s = self._duration_s
        self._cooldown_left_s = self._cooldown_s
        self._update_arrow()
        return True

    def cancel(self) -> None:
        """Stop the current push immediately, for example when the robot resets."""
        self._remaining_s = 0.0
        self._write_force(
            torch.zeros(
                1,
                self._asset.num_bodies if self._body_ids is None else len(self._body_ids),
                3,
                device=self._asset.device,
            )
        )
        if self._visualizer is not None:
            self._visualizer.set_visibility(False)

    def update(self, dt: float) -> None:
        """Advance the push timer and hide the arrow once the push ends."""
        if self._cooldown_left_s > 0.0:
            self._cooldown_left_s = max(0.0, self._cooldown_left_s - dt)

        if self._remaining_s <= 0.0:
            return
        self._remaining_s -= dt
        if self._remaining_s > 0.0:
            self._update_arrow()
            return

        self._remaining_s = 0.0
        self._write_force(
            torch.zeros(
                1,
                self._asset.num_bodies if self._body_ids is None else len(self._body_ids),
                3,
                device=self._asset.device,
            )
        )
        if self._visualizer is not None:
            self._visualizer.set_visibility(False)

    def _update_arrow(self) -> None:
        """Place the red arrow above the robot, pointing along the active push."""
        if self._visualizer is None:
            return
        root_pos = self._asset.data.root_pos_w.torch[0]
        translation = (root_pos + torch.tensor([0.0, 0.0, 0.8], device=root_pos.device)).unsqueeze(0)
        heading = torch.atan2(self._force_dir_w[1], self._force_dir_w[0]).reshape(1)
        zeros = torch.zeros_like(heading)
        orientation = quat_from_euler_xyz(zeros, zeros, heading)
        scales = torch.tensor(self._visualizer.cfg.markers["arrow"].scale, device=root_pos.device).repeat(1, 1)
        scales[:, 0] *= self._net_force * self._arrow_scale
        self._visualizer.set_visibility(True)
        self._visualizer.visualize(translations=translation, orientations=orientation, scales=scales)


def _resolve_checkpoint(
    checkpoint: str,
    task_name: str,
    agent_cfg: RslRlBaseRunnerCfg,
    agent_entry_point: str,
) -> str:
    """Resolve a checkpoint path or selector to a local checkpoint file."""
    if checkpoint in CHECKPOINT_SELECTORS:
        log_root_path = PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name
        try:
            return resolve_checkpoint_selector(
                str(log_root_path),
                checkpoint,
                library="rsl_rl",
                task=task_name,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": agent_entry_point},
            )
        except ValueError:
            # Older runs may not carry the run manifest used by the selector.
            return get_checkpoint_path(
                str(log_root_path),
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
                sort_alpha=False,
            )

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return get_checkpoint_path(str(path.parent), path.name, agent_cfg.load_checkpoint)
    return retrieve_file_path(str(path))


def _create_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg, resume_path: str):
    """Load the RSL-RL runner and return its inference policy."""
    runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        runner.load(resume_path)
    except RuntimeError as exc:
        if "size mismatch" in str(exc):
            raise RuntimeError(
                f"Checkpoint '{resume_path}' is incompatible with the current action/observation "
                "configuration. Retrain, or play a checkpoint from a matching run."
            ) from exc
        raise
    return runner.get_inference_policy(device=str(env.unwrapped.device))


def _sym_limit(bounds: tuple[float, float]) -> float:
    """Return the largest magnitude of a symmetric command range."""
    return max(abs(bounds[0]), abs(bounds[1]))


def _command_limits(term, args_cli: argparse.Namespace) -> torch.Tensor:
    """Build the per-axis command limits from the task config and CLI overrides."""
    ranges = getattr(term.cfg, "limit_ranges", None) or term.cfg.ranges
    limits = [
        args_cli.max_lin_vel_x if args_cli.max_lin_vel_x is not None else _sym_limit(ranges.lin_vel_x),
        args_cli.max_lin_vel_y if args_cli.max_lin_vel_y is not None else _sym_limit(ranges.lin_vel_y),
        args_cli.max_ang_vel_z if args_cli.max_ang_vel_z is not None else _sym_limit(ranges.ang_vel_z),
    ]
    return torch.tensor(limits, device=term.device)


def _apply_command(term, command: torch.Tensor) -> None:
    """Write the teleoperation command and keep the command manager from resampling it."""
    term.vel_command_b[0] = command
    term.time_left[0] = 1.0e9
    term.is_standing_env[0] = False


def _run_playback(
    env: RslRlVecEnvWrapper,
    policy,
    input_source: KeyboardGamepadInput,
    pusher: ForcePusher,
    args_cli: argparse.Namespace,
) -> None:
    """Run the policy while streaming keyboard/gamepad commands into the environment."""
    unwrapped = env.unwrapped
    command_term = unwrapped.command_manager.get_term(args_cli.command_term)
    step_dt = unwrapped.step_dt
    limits = _command_limits(command_term, args_cli)
    command = torch.zeros(3, device=unwrapped.device)
    smoothing = args_cli.command_smoothing
    num_steps = 0

    # The play loop owns the command buffer. Preventing resampling keeps the command stable
    # across the automatic reset that follows a fall.
    command_term._resample_command = lambda env_ids: None
    _apply_command(command_term, command)
    obs, _ = env.reset()

    print("[INFO] Playback is running, press Ctrl+C to stop.", flush=True)
    try:
        while unwrapped.sim.is_running() and not unwrapped.sim.is_stopped():
            step_start = time.time()
            state = input_source.advance()

            if state.reset:
                pusher.cancel()
                command.zero_()
                _apply_command(command_term, command)
                obs, _ = env.reset()
                policy_reset = getattr(policy, "reset", None)
                if policy_reset is not None:
                    policy_reset(torch.ones(1, dtype=torch.long, device=unwrapped.device))
                print("[INFO] Robot reset.", flush=True)
                continue

            target = torch.tensor([state.lin_x, state.lin_y, state.ang_z], device=command.device) * limits
            alpha = 1.0 - math.exp(-step_dt / smoothing) if smoothing > 0.0 else 1.0
            command += alpha * (target - command)
            _apply_command(command_term, command)

            if state.push and pusher.trigger():
                print(
                    f"[INFO] Push: {pusher.net_force:.1f} N for {pusher.duration_s:.2f} s.",
                    flush=True,
                )

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_reset = getattr(policy, "reset", None)
                if policy_reset is not None:
                    policy_reset(dones)

            if bool(dones.any()):
                pusher.cancel()
            pusher.update(step_dt)

            num_steps += 1
            if args_cli.max_steps is not None and num_steps >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Playback interrupted.", flush=True)
    finally:
        pusher.cancel()


def main() -> None:
    """Run one robot from a trained policy with interactive velocity commands."""
    parser = argparse.ArgumentParser(
        description="Play a trained Robinion policy with keyboard/gamepad control and manual force pushes.",
    )
    parser.add_argument("--task", type=str, default="RobinionGym-Velocity-Robinion", help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint path, a run directory, or 'latest'/'best'.",
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="Override the agent's experiment name.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many policy steps.")
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace playback to the policy step rate.",
    )
    parser.add_argument(
        "--train_env_cfg",
        action="store_true",
        default=False,
        help="Play with the training configuration as-is, skipping play-mode overrides.",
    )
    parser.add_argument(
        "--command-term",
        type=str,
        default="base_velocity",
        help="Name of the velocity command term to drive.",
    )
    parser.add_argument(
        "--input",
        type=str,
        choices=("keyboard", "gamepad", "both"),
        default="both",
        help="Input devices to poll. Both devices can drive the robot at once.",
    )
    parser.add_argument(
        "--gamepad-deadzone",
        type=float,
        default=0.08,
        help="Radial deadzone for the gamepad sticks.",
    )
    parser.add_argument(
        "--keyboard-gain",
        type=float,
        default=1.0,
        help="Scale factor applied to keyboard axes.",
    )
    parser.add_argument(
        "--command-smoothing",
        type=float,
        default=0.15,
        help="Low-pass time constant for command changes in seconds; 0 disables smoothing.",
    )
    parser.add_argument(
        "--max-lin-vel-x",
        type=float,
        default=None,
        help="Forward/backward command limit in m/s. Defaults to the task's trained limit.",
    )
    parser.add_argument(
        "--max-lin-vel-y",
        type=float,
        default=None,
        help="Strafe command limit in m/s. Defaults to the task's trained limit.",
    )
    parser.add_argument(
        "--max-ang-vel-z",
        type=float,
        default=None,
        help="Turn command limit in rad/s. Defaults to the task's trained limit.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=1.0e9,
        help="Episode length in seconds. The default effectively disables the time-out reset.",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=8,
        help="Render every N physics steps (400 Hz physics); 0 keeps the task setting.",
    )
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the viewport centered on the robot while it walks.",
    )
    parser.add_argument(
        "--push-force",
        type=float,
        default=None,
        help="Net horizontal push force in N. Defaults to 2.5x the robot mass.",
    )
    parser.add_argument(
        "--push-duration",
        type=float,
        default=0.2,
        help="Time the push force stays applied in seconds.",
    )
    parser.add_argument(
        "--push-cooldown",
        type=float,
        default=0.5,
        help="Time after a push before another one can be triggered in seconds.",
    )
    parser.add_argument(
        "--push-direction",
        type=str,
        choices=("forward", "backward", "left", "right", "random"),
        default="forward",
        help="Push direction relative to the robot heading.",
    )
    parser.add_argument(
        "--push-body",
        type=str,
        default=None,
        help="Optional body-name regex for the push. Defaults to every body (acceleration-style push).",
    )
    parser.add_argument(
        "--push-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a red arrow above the robot while the push is active.",
    )
    parser.add_argument(
        "--push-arrow-scale",
        type=float,
        default=0.01,
        help="Push arrow length in meters per newton.",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)

    if getattr(args_cli, "headless", False) or os.environ.get("HEADLESS") == "1":
        parser.error("Interactive play needs the Kit window; unset HEADLESS and use --viz kit.")
    # Keyboard/gamepad polling goes through the Kit application window.
    if getattr(args_cli, "visualizer", None) is None:
        args_cli.visualizer = ["kit"]
    elif "kit" not in args_cli.visualizer:
        print(
            "[WARN] Keyboard/gamepad input needs the Kit window. Add 'kit' to --viz or input will not reach the robot.",
            flush=True,
        )

    # Resolve the task configuration with play-mode overrides, exactly like `isaaclab play`.
    env_cfg, agent_cfg = resolve_task_config(
        args_cli.task,
        args_cli.agent,
        play_mode=not args_cli.train_env_cfg,
        overrides=hydra_args,
    )
    env_cfg: ManagerBasedRLEnvCfg
    agent_cfg: RslRlBaseRunnerCfg

    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    # The whole point of this script: never simulate more than one robot.
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.render_interval > 0:
        env_cfg.sim.render_interval = args_cli.render_interval
    if args_cli.follow_camera:
        # The viewport eye/lookat become offsets from the robot root instead of world points.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        # Zoom in on the single robot. The task default (8 m back, 5 m up) frames a whole
        # training scene and leaves one robot too small to read its pose while driving it.
        env_cfg.viewer.eye = (3.0, 0.0, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.6)

    # Play mode keeps the automatic push event at its trained strength. Replace it with the
    # manual push so the robot is only shoved when the user asks for it.
    events = getattr(env_cfg, "events", None)
    if events is not None and hasattr(events, "push_robot"):
        events.push_robot = None

    task_name = args_cli.task.split(":")[-1]
    resume_path = _resolve_checkpoint(args_cli.checkpoint, task_name, agent_cfg, args_cli.agent)
    print(f"[INFO] Task: {task_name}", flush=True)
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}", flush=True)
    print(f"[INFO] Checkpoint: {resume_path}", flush=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_name, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policy = _create_runner(env, agent_cfg, resume_path)

        input_source = KeyboardGamepadInput(
            mode=args_cli.input,
            deadzone=args_cli.gamepad_deadzone,
            keyboard_gain=args_cli.keyboard_gain,
        )
        robot: Articulation = env.unwrapped.scene["robot"]
        pusher = ForcePusher(
            asset=robot,
            force=args_cli.push_force,
            duration_s=args_cli.push_duration,
            cooldown_s=args_cli.push_cooldown,
            direction=args_cli.push_direction,
            body_expr=args_cli.push_body,
            visualize=args_cli.push_vis,
            arrow_scale=args_cli.push_arrow_scale,
        )

        print(
            "[INFO] Controls: W/S forward/back, A/D turn, Q/E strafe, F push, R reset. "
            "Gamepad: left stick move, right stick turn, A push, X reset.",
            flush=True,
        )
        try:
            _run_playback(env, policy, input_source, pusher, args_cli)
        finally:
            env.close()


if __name__ == "__main__":
    main()
