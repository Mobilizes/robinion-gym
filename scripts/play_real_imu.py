# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained Robinion policy in mid-air while a real IMU drives its observations.

The robot is spawned high above the ground with its root position pinned, so the legs can run
through the walking gait without the robot falling or translating. The root orientation follows
the physical IMU, so tilting or rotating the board also tilts and rotates the whole robot. The
trained policy still receives the base angular velocity and projected-gravity terms it was
trained with, but both terms are replaced with values measured by the physical IMU in
``imu_read/``:

* ``projected_gravity`` comes from the IMU quaternion. The first valid sample is captured as the
  neutral orientation (disable it with ``--no-imu-zero``) so a board that is not perfectly level
  at startup still tells the policy it is upright.
* ``base_ang_vel`` is the IMU's body-frame angular velocity, estimated by finite-differencing
  consecutive orientation samples. Disable it with ``--no-imu-ang-vel`` to keep the simulated
  (zero) root angular velocity instead.

Tilt or rotate the physical board while the demo runs and the policy reacts as if the robot's
base were being disturbed. If no IMU is connected, the demo falls back to the simulated base
observations and keeps running so it can still be tested.

.. code-block:: bash

    # Latest fully trained run in this checkout; robot hangs at 1.5 m
    ./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1

    # Explicit board port, larger walking command, headless smoke test
    HEADLESS=1 ./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1 \
        --imu-port /dev/ttyUSB0 --command-lin-x 0.6 --max_steps 100

    # Disable the finite-difference gyro and only feed orientation
    ./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1 --no-imu-ang-vel
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)

from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

import robinion_gym.tasks  # noqa: F401

PROJECT_DIR = Path(__file__).resolve().parents[1]

#: Default location of the sibling ``imu_read`` checkout that owns the serial driver.
DEFAULT_IMU_DIR = PROJECT_DIR.parent / "imu_read"

#: World-frame gravity direction used to build the projected-gravity observation.
_GRAVITY_W = (0.0, 0.0, -1.0)

#: Live source shared by the observation terms and the telemetry printer. Manager configs are
#: deep-copied when managers are created, which would otherwise duplicate a bound-method source
#: and leave the printed status reading a stale copy.
_IMU_SOURCE: RealImuObservation | None = None


def _find_imu_dir() -> Path:
    """Return the directory that contains the ``imu_bridge`` package."""
    candidates: list[Path] = []
    env_dir = os.environ.get("IMU_READ_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates += [
        DEFAULT_IMU_DIR,
        PROJECT_DIR / "imu_read",
        Path.cwd() / "imu_read",
        Path.cwd().parent / "imu_read",
    ]
    for candidate in candidates:
        if (candidate / "imu_bridge" / "serial_imu.py").is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Could not find the imu_read checkout (expected an 'imu_bridge/serial_imu.py' file). "
        f"Searched:\n  {searched}\nSet IMU_READ_DIR to the checkout that contains 'imu_bridge'."
    )


class RealImuObservation:
    """Translate the physical IMU orientation into the policy's base observations.

    The serial sensor streams fused orientation only (no accelerometer or gyroscope), so the
    angular velocity is reconstructed from successive quaternions. The source is shared by the
    observation terms so both see the same sample and the finite difference advances only once
    per environment step.
    """

    def __init__(
        self,
        sensor_name: str,
        zero_on_start: bool,
        use_ang_vel: bool,
        ang_vel_filter_s: float,
        ang_vel_hold_s: float,
        ang_vel_deadband: float,
        ang_vel_max: float,
    ):
        """Create the source.

        Args:
            sensor_name: Name of the ``SerialImu`` sensor in the scene.
            zero_on_start: Treat the first valid sample as the robot's neutral orientation.
            use_ang_vel: Replace the base angular-velocity observation with the IMU estimate.
            ang_vel_filter_s: Low-pass time constant for the finite-difference angular velocity.
            ang_vel_hold_s: Time constant used to decay the estimate while no new sample arrives.
            ang_vel_deadband: Angular-speed magnitude below which the estimate is treated as zero.
            ang_vel_max: Maximum angular-speed magnitude in rad/s after differentiation.
        """
        self._sensor_name = sensor_name
        self._zero_on_start = zero_on_start
        self._use_ang_vel = use_ang_vel
        self._ang_vel_filter_s = max(ang_vel_filter_s, 1.0e-4)
        self._ang_vel_hold_s = max(ang_vel_hold_s, 1.0e-4)
        self._ang_vel_deadband = max(ang_vel_deadband, 0.0)
        self._ang_vel_max = max(ang_vel_max, 0.0)

        self._gravity_w = torch.tensor(_GRAVITY_W, dtype=torch.float32)
        self._reference_quat: torch.Tensor | None = None
        self._last_quat: torch.Tensor | None = None
        self._last_timestamp: float | None = None
        self._last_seq: int | None = None
        self._last_obs_step: int | None = None
        self._gravity: torch.Tensor | None = None
        self._ang_vel: torch.Tensor | None = None
        self._orientation_xyzw: torch.Tensor | None = None

        self.valid = False
        self.sample_count = 0
        self.reference_captured = False
        self.sensor_port: str | None = None
        self.last_error: str | None = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.sample_timestamp = 0.0

    @staticmethod
    def _to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Convert a ``(w, x, y, z)`` quaternion to Isaac Lab's ``(x, y, z, w)`` order."""
        return quat_wxyz[..., [1, 2, 3, 0]]

    def _sensor(self, env):
        return env.scene[self._sensor_name]

    def poll(self, env, dt: float = 0.01) -> None:
        """Force a fresh sensor read while the simulation is not stepping (startup wait)."""
        self._sensor(env).update(dt, force_recompute=True)
        self._refresh(env, force=True)

    def _observation_step(self, env) -> int | None:
        """Return the current environment-step counter, if the environment exposes one."""
        episode_length = getattr(env, "episode_length_buf", None)
        if episode_length is not None:
            return int(episode_length[0].item())
        return None

    def _refresh(self, env, force: bool = False) -> None:
        """Read one sensor sample and update the cached observation values once per env step.

        Args:
            env: The manager-based environment.
            force: Read the sensor even if this environment step was already sampled. Used while
                waiting for the hardware before the simulation loop starts.
        """
        step = self._observation_step(env)
        if not force and step is not None and step == self._last_obs_step:
            return
        self._last_obs_step = step

        sensor = self._sensor(env)
        data = sensor.data
        self.sensor_port = getattr(sensor, "serial_port", None)
        self.last_error = getattr(getattr(sensor, "_reader", None), "last_error", None)
        if not bool(data.valid[0]):
            self.valid = False
            # The base is locked in the air, so the last estimate is the best guess for a short
            # dropout. Let the angular velocity decay rather than latching a stale rotation rate.
            if self._ang_vel is not None:
                decay = math.exp(-env.step_dt / self._ang_vel_hold_s)
                self._ang_vel = self._ang_vel * decay
            return

        # ``seq`` is the reliable new-sample marker. Timestamps are stored as float64 in the
        # current bridge, but comparing quaternions keeps this working with older sensor builds
        # whose timestamp tensor was float32 (and therefore unusable at Unix-epoch magnitude).
        seq_tensor = getattr(data, "seq", None)
        seq = int(seq_tensor[0]) if seq_tensor is not None else None
        quat_wxyz = data.quat_w[0].to(dtype=torch.float32)
        quat = self._to_xyzw(quat_wxyz)
        timestamp = float(data.timestamp[0])
        same_sample = (seq is not None and seq == self._last_seq) or (
            seq is None and self._last_quat is not None and torch.equal(quat, self._last_quat)
        )
        if not force and same_sample:
            # The reader has not produced a new sample for this environment step. Decay a held
            # gyro estimate so a stalled stream cannot look like a continuous rotation.
            if self._use_ang_vel and self._ang_vel is not None:
                decay = math.exp(-env.step_dt / self._ang_vel_hold_s)
                self._ang_vel = self._ang_vel * decay
            self.valid = True
            return

        self.valid = True
        self.sample_count += 1
        self.sample_timestamp = timestamp
        self._last_seq = seq
        self.yaw = float(data.yaw[0])
        self.pitch = float(data.pitch[0])
        self.roll = float(data.roll[0])
        if self._reference_quat is None:
            # Capture the startup pose so the demo is independent of how level the desk is. With
            # zeroing disabled the world frame is used directly.
            if self._zero_on_start:
                self._reference_quat = quat.clone()
                self.reference_captured = True
            else:
                self._reference_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat.device)
        quat_rel = quat_mul(quat_conjugate(self._reference_quat), quat)
        self._orientation_xyzw = quat_rel
        gravity_w = self._gravity_w.to(device=quat.device)
        self._gravity = quat_apply_inverse(quat_rel, gravity_w)

        if not self._use_ang_vel:
            self._last_quat = quat.clone()
            self._last_timestamp = timestamp
            return

        if self._last_quat is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if 1.0e-4 < dt < 0.25:
                delta = quat_mul(quat_conjugate(self._last_quat), quat)
                raw = axis_angle_from_quat(delta.unsqueeze(0))[0] / dt
                magnitude = torch.linalg.norm(raw)
                if magnitude < self._ang_vel_deadband:
                    raw = torch.zeros_like(raw)
                elif magnitude > self._ang_vel_max:
                    # A reconnect can jump the orientation between samples. Do not feed the
                    # resulting impulse into the policy.
                    raw = raw * (self._ang_vel_max / magnitude)
                if self._ang_vel is None:
                    self._ang_vel = raw
                else:
                    alpha = 1.0 - math.exp(-dt / self._ang_vel_filter_s)
                    self._ang_vel = self._ang_vel + alpha * (raw - self._ang_vel)
            elif dt >= 0.25:
                # The stream was interrupted; do not differentiate across the gap.
                self._ang_vel = torch.zeros(3, device=quat.device)

        self._last_quat = quat.clone()
        self._last_timestamp = timestamp

    def projected_gravity(self, env) -> torch.Tensor:
        """Observation term: gravity direction in the (IMU-measured) base frame."""
        self._refresh(env)
        if self._gravity is None:
            return env.scene["robot"].data.projected_gravity_b.torch
        return self._gravity.unsqueeze(0).expand(env.num_envs, -1)

    def base_ang_vel(self, env) -> torch.Tensor:
        """Observation term: angular velocity in the (IMU-measured) base frame."""
        self._refresh(env)
        if self._ang_vel is None:
            return torch.zeros(env.num_envs, 3, device=env.device)
        return self._ang_vel.unsqueeze(0).expand(env.num_envs, -1)

    def orientation_xyzw(self, env) -> torch.Tensor:
        """Return the IMU orientation relative to startup, in ``(x, y, z, w)`` order."""
        self._refresh(env)
        if self._orientation_xyzw is None:
            return torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        return self._orientation_xyzw

    def status(self) -> str:
        """Return a one-line human-readable summary of the latest sample."""
        if not self.valid:
            port = self.sensor_port or "auto-detect"
            reason = self.last_error or "sensor invalid/stale"
            return f"[IMU] waiting for samples on {port} ({reason})"
        gravity = self._gravity.tolist() if self._gravity is not None else [float("nan")] * 3
        ang_vel = self._ang_vel.tolist() if self._ang_vel is not None else [0.0, 0.0, 0.0]
        return (
            f"[IMU] {self.sensor_port or 'auto-detect'} valid={self.valid} "
            f"yaw={self.yaw:8.2f} pitch={self.pitch:7.2f} roll={self.roll:7.2f}  "
            f"gravity_b=({gravity[0]:+.3f}, {gravity[1]:+.3f}, {gravity[2]:+.3f})  "
            f"ang_vel_b=({ang_vel[0]:+.2f}, {ang_vel[1]:+.2f}, {ang_vel[2]:+.2f}) rad/s"
        )


def _imu_projected_gravity(env) -> torch.Tensor:
    """Observation term that reads the shared live IMU source."""
    if _IMU_SOURCE is None:
        raise RuntimeError("The real IMU source is not initialized.")
    return _IMU_SOURCE.projected_gravity(env)


def _imu_base_ang_vel(env) -> torch.Tensor:
    """Observation term that reads the shared live IMU source."""
    if _IMU_SOURCE is None:
        raise RuntimeError("The real IMU source is not initialized.")
    return _IMU_SOURCE.base_ang_vel(env)


def _install_imu_observations(env_cfg: ManagerBasedRLEnvCfg, use_ang_vel: bool) -> None:
    """Replace the simulated base observations with the real IMU values."""
    groups = [env_cfg.observations.actor]
    critic = getattr(env_cfg.observations, "critic", None)
    if critic is not None:
        groups.append(critic)

    for group in groups:
        if not hasattr(group, "projected_gravity"):
            raise RuntimeError(
                "The task's observation group has no 'projected_gravity' term; the real IMU demo "
                "expects the Robinion velocity task."
            )
        group.projected_gravity.func = _imu_projected_gravity
        group.projected_gravity.noise = None
        if use_ang_vel:
            if not hasattr(group, "base_ang_vel"):
                raise RuntimeError("The task's observation group has no 'base_ang_vel' term.")
            group.base_ang_vel.func = _imu_base_ang_vel
            group.base_ang_vel.noise = None


def _setup_air_robot(env_cfg: ManagerBasedRLEnvCfg, height: float) -> None:
    """Spawn the robot at *height* with an unpinned root.

    The root position is pinned and the root orientation is driven from the IMU each policy
    step in :func:`_run_demo`. The task's robot config uses the deprecated merged
    articulation-root cfg; with Newton, its ``fix_root_link`` path relocates the articulation
    root and can fail on assets whose Newton properties live on a child link, so replace it with
    the current fragment form and explicitly leave the root unpinned.
    """
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg

    robot_cfg = env_cfg.scene.robot
    robot_cfg.init_state.pos = (0.0, 0.0, height)

    spawn = robot_cfg.spawn
    spawn.articulation_props = [
        PhysxArticulationCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        NewtonArticulationCfg(self_collision_enabled=True),
    ]
    spawn.fix_root_link = False


def _add_imu_sensor(env_cfg: ManagerBasedRLEnvCfg, args_cli: argparse.Namespace) -> Path:
    """Add the physical IMU as a scene sensor before the environment is built.

    Returns:
        The ``imu_read`` checkout that provided the sensor.
    """
    imu_dir = _find_imu_dir()
    if str(imu_dir) not in sys.path:
        sys.path.insert(0, str(imu_dir))
    try:
        from imu_bridge.serial_imu import SerialImuCfg
    except ModuleNotFoundError as exc:
        if exc.name == "serial":
            raise RuntimeError(
                "pyserial is required for the real IMU demo. Install it into the project "
                "environment with: uv pip install --python .venv/bin/python pyserial"
            ) from exc
        raise

    imu_cfg = SerialImuCfg(
        prim_path="/World/SerialImu",
        port=args_cli.imu_port,
        baud=args_cli.imu_baud,
        stale_timeout=args_cli.imu_stale_timeout,
    )
    # A global prim path yields one hardware sample, which is broadcast to the single demo env.
    # InteractiveScene discovers assets from the config's __dict__, so this adds the extra sensor
    # without changing the registered task configuration class.
    env_cfg.scene.__dict__["imu"] = imu_cfg
    return imu_dir


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
    if not path.exists():
        # Allow a bare run directory name such as "walk-v6.1" in addition to a full path.
        run_dir = PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name / path
        if run_dir.is_dir():
            path = run_dir
    if path.is_dir():
        # Anchor the run name: get_checkpoint_path() matches it with re.match(), so "walk-v6.1"
        # would also select the sibling "walk-v6.1_" run.
        return get_checkpoint_path(str(path.parent), f"^{re.escape(path.name)}$", agent_cfg.load_checkpoint)
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


def _apply_command(term, command: torch.Tensor) -> None:
    """Write a constant velocity command and keep the command manager from resampling it."""
    term.vel_command_b[0] = command
    term.time_left[0] = 1.0e9
    term.is_standing_env[0] = False


def _wait_for_imu(source: RealImuObservation, env, timeout_s: float) -> None:
    """Wait briefly for the first valid hardware sample so the demo starts with real data."""
    if timeout_s <= 0.0:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        source.poll(env)
        if source.valid:
            print(f"[INFO] IMU connected: {source.sensor_port}", flush=True)
            if source.reference_captured:
                print("[INFO] Captured the startup orientation as the neutral pose.", flush=True)
            return
        time.sleep(0.05)
    print(
        "[WARN] No valid IMU sample yet; the demo keeps running with the simulated base "
        "observations until the sensor starts streaming.",
        flush=True,
    )


def _run_demo(
    env: RslRlVecEnvWrapper,
    policy,
    source: RealImuObservation,
    args_cli: argparse.Namespace,
) -> None:
    """Step the policy with a constant command while the real IMU drives the base observations."""
    unwrapped = env.unwrapped
    command_term = unwrapped.command_manager.get_term(args_cli.command_term)
    command = torch.tensor(
        [args_cli.command_lin_x, args_cli.command_lin_y, args_cli.command_ang_z],
        device=unwrapped.device,
    )
    command_term._resample_command = lambda env_ids: None
    _apply_command(command_term, command)
    obs, _ = env.reset()
    _apply_command(command_term, command)

    robot = unwrapped.scene["robot"]
    root_position_w = robot.data.root_pos_w.torch[0].clone()
    upright_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=unwrapped.device)
    zero_root_velocity = torch.zeros(1, 6, device=unwrapped.device)

    def target_base_quat() -> torch.Tensor:
        """Return the base quaternion that :func:`sync_base` writes."""
        return source.orientation_xyzw(unwrapped) if args_cli.base_follow_imu else upright_quat

    def sync_base() -> None:
        """Pin the root position and optionally rotate the base to the IMU orientation."""
        root_pose = torch.cat((root_position_w, target_base_quat())).unsqueeze(0)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=zero_root_velocity)

    sync_base()
    step_dt = unwrapped.step_dt
    last_print = 0.0
    num_steps = 0
    print(
        f"[INFO] Air-walking with command (vx={args_cli.command_lin_x:.2f} m/s, "
        f"vy={args_cli.command_lin_y:.2f} m/s, wz={args_cli.command_ang_z:.2f} rad/s). "
        "Tilt or rotate the physical IMU to disturb the policy.",
        flush=True,
    )

    try:
        while unwrapped.sim.is_running() and not unwrapped.sim.is_stopped():
            step_start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_reset = getattr(policy, "reset", None)
                if policy_reset is not None:
                    policy_reset(dones)

            # Hold the root at the spawn point and rotate the whole robot to the IMU pose for
            # the next physics step. The policy's base observations are the same IMU values.
            sync_base()

            if bool(dones.any()):
                # The command manager resets with the environment; restore the constant command.
                _apply_command(command_term, command)

            num_steps += 1
            now = time.time()
            if args_cli.print_interval > 0.0 and now - last_print >= args_cli.print_interval:
                last_print = now
                print(source.status(), flush=True)
                try:
                    obs_vector = obs["actor"].reshape(-1)
                except (KeyError, TypeError, IndexError):
                    obs_vector = obs.reshape(-1) if isinstance(obs, torch.Tensor) else None
                if obs_vector is not None:
                    root_z = float(robot.data.root_pos_w.torch[0, 2])
                    joint_speed = float(robot.data.joint_vel.torch[0].norm())
                    base_roll, base_pitch, base_yaw = euler_xyz_from_quat(target_base_quat().unsqueeze(0))
                    print(
                        f"[ROBOT] root_z={root_z:.3f} m |qdot|={joint_speed:.2f} rad/s  "
                        f"base_rpy=({float(base_roll[0]):+.2f}, {float(base_pitch[0]):+.2f}, "
                        f"{float(base_yaw[0]):+.2f}) rad  "
                        f"obs_ang_vel={obs_vector[0:3].tolist()}  "
                        f"obs_gravity={obs_vector[3:6].tolist()}",
                        flush=True,
                    )

            if args_cli.max_steps is not None and num_steps >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Demo interrupted.", flush=True)


def main() -> None:
    """Run the mid-air robot with real IMU observations."""
    global _IMU_SOURCE

    parser = argparse.ArgumentParser(
        description="Run a trained Robinion policy in mid-air with real IMU observations.",
        epilog=(
            "The imu_bridge package is loaded from IMU_READ_DIR, the sibling '../imu_read' checkout, or './imu_read'."
        ),
    )
    parser.add_argument("--task", type=str, default="RobinionGym-Velocity-Robinion", help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="walk-v6.1",
        help=(
            "Checkpoint path, a run directory, or 'latest'/'best'. Defaults to the latest fully "
            "trained run in this checkout; pass a path to use another policy."
        ),
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
    parser.add_argument("--command-lin-x", type=float, default=0.4, help="Forward velocity command in m/s.")
    parser.add_argument("--command-lin-y", type=float, default=0.0, help="Strafe velocity command in m/s.")
    parser.add_argument("--command-ang-z", type=float, default=0.0, help="Turn velocity command in rad/s.")
    parser.add_argument(
        "--height",
        type=float,
        default=1.5,
        help="Root height in meters where the robot is locked in the air.",
    )
    parser.add_argument(
        "--base-follow-imu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate the robot base to the IMU orientation, in addition to feeding the observations.",
    )
    parser.add_argument(
        "--imu-port",
        type=str,
        default=None,
        help="IMU serial port. Defaults to auto-detection (CH340) or IMU_PORT.",
    )
    parser.add_argument("--imu-baud", type=int, default=115200, help="IMU serial baud rate.")
    parser.add_argument(
        "--imu-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat the first IMU sample as the robot's neutral orientation.",
    )
    parser.add_argument(
        "--imu-ang-vel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed the finite-difference IMU angular velocity into the base_ang_vel observation.",
    )
    parser.add_argument(
        "--imu-ang-vel-filter",
        type=float,
        default=0.05,
        help="Low-pass time constant for the IMU angular velocity in seconds.",
    )
    parser.add_argument(
        "--imu-ang-vel-hold",
        type=float,
        default=0.1,
        help="Time constant for decaying the angular velocity while samples are stale, in seconds.",
    )
    parser.add_argument(
        "--imu-ang-vel-deadband",
        type=float,
        default=0.02,
        help="Angular speed in rad/s below which the IMU estimate is treated as zero.",
    )
    parser.add_argument(
        "--imu-ang-vel-max",
        type=float,
        default=20.0,
        help="Maximum reconstructed angular speed in rad/s, guarding against reconnect jumps.",
    )
    parser.add_argument(
        "--imu-stale-timeout",
        type=float,
        default=0.5,
        help="Samples older than this are treated as invalid in seconds.",
    )
    parser.add_argument(
        "--imu-wait",
        type=float,
        default=2.0,
        help="Seconds to wait for the first valid IMU sample before falling back to simulation.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between IMU status prints; 0 disables them.",
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
        help="Use a close fixed world-space view of the robot.",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)

    headless_env = os.environ.get("HEADLESS", "0") == "1"
    if getattr(args_cli, "visualizer", None) is None and not headless_env:
        args_cli.visualizer = ["kit"]

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

    # One robot, hanging in the air. Its root position is held at the spawn point in the play
    # loop, while its root orientation follows the real IMU.
    env_cfg.scene.num_envs = 1
    _setup_air_robot(env_cfg, args_cli.height)
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.render_interval > 0:
        env_cfg.sim.render_interval = args_cli.render_interval
    if args_cli.follow_camera:
        # Fixed world-space view: the robot stays at one point, so tracking its root would
        # make the whole scene spin with the IMU-driven body rotation.
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (3.0, 0.0, args_cli.height + 0.7)
        env_cfg.viewer.lookat = (0.0, 0.0, args_cli.height)

    events = getattr(env_cfg, "events", None)
    if events is not None:
        # No random root pose or velocity: the robot must stay at the spawn pose so the fixed
        # joint never fights a teleported root. Keep joint/actuator randomization.
        if hasattr(events, "reset_all"):
            events.reset_all = None
        if hasattr(events, "push_robot"):
            events.push_robot = None
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None:
        # Leg/self collisions are expected while walking in the air and must not reset the demo.
        if hasattr(terminations, "term_upper_body_collision"):
            terminations.term_upper_body_collision = None
        if hasattr(terminations, "term_leg_leg_collision"):
            terminations.term_leg_leg_collision = None
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None and hasattr(rewards, "pen_self_collision"):
        # This reward term references the two disabled terminations above. Rewards are not used
        # during playback, so drop the term rather than leaving a dangling reference.
        rewards.pen_self_collision = None

    imu_dir = _add_imu_sensor(env_cfg, args_cli)
    source = RealImuObservation(
        sensor_name="imu",
        zero_on_start=args_cli.imu_zero,
        use_ang_vel=args_cli.imu_ang_vel,
        ang_vel_filter_s=args_cli.imu_ang_vel_filter,
        ang_vel_hold_s=args_cli.imu_ang_vel_hold,
        ang_vel_deadband=args_cli.imu_ang_vel_deadband,
        ang_vel_max=args_cli.imu_ang_vel_max,
    )
    _IMU_SOURCE = source
    _install_imu_observations(env_cfg, args_cli.imu_ang_vel)

    task_name = args_cli.task.split(":")[-1]
    resume_path = _resolve_checkpoint(args_cli.checkpoint, task_name, agent_cfg, args_cli.agent)
    print(f"[INFO] Task: {task_name}", flush=True)
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}", flush=True)
    print(f"[INFO] Checkpoint: {resume_path}", flush=True)
    print(f"[INFO] IMU package: {imu_dir}", flush=True)
    base_behavior = "follows the IMU" if args_cli.base_follow_imu else "stays upright"
    print(
        f"[INFO] Robot position pinned at z={args_cli.height:.2f} m; base orientation {base_behavior}.",
        flush=True,
    )
    print(
        "[INFO] Real IMU replaces actor observations: projected_gravity"
        + (" and base_ang_vel." if args_cli.imu_ang_vel else "."),
        flush=True,
    )

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_name, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policy = _create_runner(env, agent_cfg, resume_path)

        _wait_for_imu(source, env.unwrapped, args_cli.imu_wait)
        try:
            _run_demo(env, policy, source, args_cli)
        finally:
            # Stop the serial reader explicitly so the port is released before the process exits.
            with contextlib.suppress(Exception):
                env.unwrapped.scene["imu"].close()
            env.close()


if __name__ == "__main__":
    main()
