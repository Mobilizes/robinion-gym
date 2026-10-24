# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Robinion policy in one environment and publish its actions over ROS 2.

The script mirrors the ``isaaclab play`` workflow for the registered ``RobinionGym-Velocity-Robinion``
task, but forces ``num_envs=1``. Every policy step it publishes the commands that were applied to
that robot:

* ``/robinion/joint_commands`` (``sensor_msgs/msg/JointState``): processed joint position targets
  in radians, with the resolved joint names.
* ``/robinion/policy_actions`` (``std_msgs/msg/Float64MultiArray``): raw policy outputs after the
  RSL-RL action clip.

When no sourced ROS 2 Python environment is available, the script uses the ROS 2 Humble/Jazzy
runtime bundled with Isaac Sim and restarts itself with the matching environment variables. This is
needed because the system ROS 2 Humble installation targets Python 3.10, while Isaac Lab runs in
Python 3.12.

.. code-block:: bash

    # Latest checkpoint, run as fast as possible
    ./isaaclab.sh -p scripts/play_ros2.py --checkpoint latest

    # Specific checkpoint at roughly the policy rate
    ./isaaclab.sh -p scripts/play_ros2.py \
        --checkpoint logs/rsl_rl/robinion_velocity/walk-v5/model_9999.pt --real-time

    # Inspect the commands from another terminal (system ROS 2 Humble)
    ros2 topic echo /robinion/joint_commands
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

_ROS2_BOOTSTRAP_ENV_VAR = "_ROBINION_PLAY_ROS2_READY"
_ROS2_PREFIX_ENV_VAR = "_ROBINION_PLAY_ROS2_PREFIX"
_ROS2_PYTHONPATH_ENV_VAR = "_ROBINION_PLAY_ROS2_PYTHONPATH"

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read_cli_option(name: str) -> str | None:
    """Return the value of a ``--name value`` or ``--name=value`` argument from ``sys.argv``."""
    argv = sys.argv[1:]
    for index, argument in enumerate(argv):
        if argument == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _prepend_path(current: str | None, path: Path) -> str:
    """Prepend *path* to a colon-separated environment variable value."""
    return f"{path}{os.pathsep}{current}" if current else str(path)


def _rclpy_is_importable() -> bool:
    """Check ``import rclpy`` in a child process without polluting this interpreter."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import rclpy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _record_rclpy_python_path() -> None:
    """Store the directory containing the importable ``rclpy`` package for later use."""
    spec = importlib.util.find_spec("rclpy")
    if spec is not None and spec.origin is not None:
        os.environ[_ROS2_PYTHONPATH_ENV_VAR] = str(Path(spec.origin).resolve().parent.parent)


def _add_ros2_python_paths() -> None:
    """Re-add the runtime's Python path after Kit has started."""
    prefix = os.environ.get(_ROS2_PREFIX_ENV_VAR)
    if prefix is not None:
        candidate = Path(prefix) / "rclpy"
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    python_path = os.environ.get(_ROS2_PYTHONPATH_ENV_VAR)
    if python_path is not None and python_path not in sys.path:
        sys.path.insert(0, python_path)


def _find_bundled_ros2_prefix(requested: str) -> Path | None:
    """Locate the ROS 2 runtime bundled inside Isaac Sim."""
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        return None
    package_root = Path(spec.origin).resolve().parent

    candidates: list[str] = []
    if requested == "auto":
        # Match the distribution Isaac Sim itself selects for the host Ubuntu release.
        try:
            version = platform.freedesktop_os_release().get("VERSION_ID", "")
        except OSError:
            version = ""
        preferred = {"22": "humble", "24": "jazzy"}.get(version.split(".")[0] if version else "")
        if preferred is not None:
            candidates.append(preferred)
        candidates.extend(["humble", "jazzy"])
    else:
        candidates.append(requested)

    for distro in dict.fromkeys(candidates):
        prefix = package_root / "exts" / "isaacsim.ros2.core" / distro
        if (prefix / "rclpy" / "rclpy" / "__init__.py").is_file():
            return prefix
    return None


def _bootstrap_ros2_runtime() -> None:
    """Ensure ``rclpy`` is importable, restarting the process with the bundled runtime if needed."""
    if os.environ.get(_ROS2_BOOTSTRAP_ENV_VAR) == "1":
        return
    # Help and ROS-free playback do not need a ROS 2 runtime.
    if "--no-ros" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
        return

    requested = _read_cli_option("--ros-distro") or "auto"
    if requested == "system":
        if not _rclpy_is_importable():
            raise RuntimeError(
                "--ros-distro system was requested, but 'import rclpy' failed. "
                "Source a ROS 2 installation built for this Python interpreter first."
            )
        _record_rclpy_python_path()
        return

    # Reuse an already sourced ROS 2 environment when it matches the requested distribution.
    if _rclpy_is_importable() and (requested == "auto" or os.environ.get("ROS_DISTRO") == requested):
        _record_rclpy_python_path()
        return

    prefix = _find_bundled_ros2_prefix(requested)
    if prefix is None:
        raise RuntimeError(
            "No ROS 2 Python runtime was found. Source a ROS 2 installation built for this Python "
            "interpreter and pass --ros-distro system, or install the Isaac Sim ROS 2 extension."
        )

    distro = prefix.name
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(env.get("PYTHONPATH"), prefix / "rclpy")
    library_path_var = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    env[library_path_var] = _prepend_path(env.get(library_path_var), prefix / "lib")
    env["AMENT_PREFIX_PATH"] = _prepend_path(env.get("AMENT_PREFIX_PATH"), prefix)
    env["ROS_DISTRO"] = distro
    env.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    env[_ROS2_BOOTSTRAP_ENV_VAR] = "1"
    env[_ROS2_PREFIX_ENV_VAR] = str(prefix)

    print(f"[INFO] Restarting with the ROS 2 {distro} runtime bundled with Isaac Sim: {prefix}")
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)


_bootstrap_ros2_runtime()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli  # noqa: E402

import robinion_gym.tasks  # noqa: F401,E402


class RosActionPublisher:
    """Publish one robot's commands and raw policy actions to ROS 2."""

    def __init__(
        self,
        node_name: str,
        joint_topic: str,
        action_topic: str | None,
        joint_names: list[str],
    ) -> None:
        """Create the ROS 2 node and publishers.

        Args:
            node_name: Name of the ROS 2 node.
            joint_topic: Topic for the processed joint position targets.
            action_topic: Optional topic for the raw policy actions.
            joint_names: Joint names in the same order as the action term.
        """
        # Add the runtime again in case Kit rewrote sys.path during startup.
        _add_ros2_python_paths()

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self._rclpy = rclpy
        self._joint_state_type = JointState
        self._float64_multi_array_type = Float64MultiArray
        self._joint_names = joint_names

        rclpy.init(args=None)
        self._node = Node(node_name)
        self._joint_publisher = self._node.create_publisher(JointState, joint_topic, 10)
        self._action_publisher = None
        if action_topic is not None:
            self._action_publisher = self._node.create_publisher(Float64MultiArray, action_topic, 10)

        published = f"{len(joint_names)} joint targets on '{joint_topic}'"
        if action_topic is not None:
            published += f" and raw actions on '{action_topic}'"
        runtime = (
            f"ROS_DISTRO={os.environ.get('ROS_DISTRO', 'system')}, "
            f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}, "
            f"ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '0')}, "
            f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', 'default')}"
        )
        self._node.get_logger().info(f"Publishing {published} ({runtime})")

    def publish(self, joint_positions: torch.Tensor, actions: torch.Tensor) -> None:
        """Publish one command pair.

        Args:
            joint_positions: Processed joint position targets, shape ``(num_joints,)``.
            actions: Raw policy actions, shape ``(num_joints,)``.
        """
        stamp = self._node.get_clock().now().to_msg()

        joint_message = self._joint_state_type()
        joint_message.header.stamp = stamp
        joint_message.name = self._joint_names
        joint_message.position = joint_positions.detach().cpu().tolist()
        self._joint_publisher.publish(joint_message)

        if self._action_publisher is not None:
            action_message = self._float64_multi_array_type()
            action_message.data = actions.detach().cpu().tolist()
            self._action_publisher.publish(action_message)

    def close(self) -> None:
        """Destroy the node and shut down the ROS 2 context."""
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


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


def _create_runner(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
):
    """Load the RSL-RL runner and return its inference policy."""
    runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        runner.load(resume_path)
    except RuntimeError as exc:
        if "size mismatch" in str(exc):
            raise RuntimeError(
                f"Checkpoint '{resume_path}' is incompatible with the current 23-action/94-observation "
                "configuration. The passive-bar change removed four actions and four last-action "
                "observations; retrain before playing this checkpoint."
            ) from exc
        raise
    return runner.get_inference_policy(device=str(env.unwrapped.device))


def _run_playback(
    env: RslRlVecEnvWrapper,
    policy,
    publisher: RosActionPublisher | None,
    args_cli: argparse.Namespace,
) -> None:
    """Step the environment and publish the resulting commands until interrupted."""
    action_term = env.unwrapped.action_manager.get_term(args_cli.action_term)
    step_dt = env.unwrapped.step_dt
    obs = env.get_observations()
    num_steps = 0

    print("[INFO] Playback is running, press Ctrl+C to stop.", flush=True)
    try:
        while True:
            step_start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                # Mirror the wrapper's clip so the published values are the ones sent to the env,
                # including on the step that resets the action history.
                if env.clip_actions is not None:
                    actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
                applied_actions = actions[0].clone()
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            if publisher is not None:
                publisher.publish(action_term.processed_actions[0], applied_actions)

            num_steps += 1
            if args_cli.max_steps is not None and num_steps >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Playback interrupted.", flush=True)


def main() -> None:
    """Run one environment of the trained policy and publish its commands over ROS 2."""
    parser = argparse.ArgumentParser(
        description="Play a trained Robinion policy in one environment and publish its commands over ROS 2.",
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
    parser.add_argument(
        "--real-time", action="store_true", default=False, help="Pace playback to the policy step rate."
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many policy steps.")
    parser.add_argument(
        "--train_env_cfg",
        action="store_true",
        default=False,
        help="Play with the training configuration as-is, skipping play-mode overrides.",
    )
    parser.add_argument("--action_term", type=str, default="joint_pos", help="Action term to publish.")
    parser.add_argument(
        "--ros-distro",
        type=str,
        choices=("auto", "humble", "jazzy", "system"),
        default="auto",
        help="ROS 2 runtime to use. 'auto' prefers a sourced environment, then Isaac Sim's bundled runtime.",
    )
    parser.add_argument("--ros-node", type=str, default="robinion_play", help="Name of the ROS 2 node.")
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help="Override ROS_DOMAIN_ID for this node; the subscribing shell must use the same value.",
    )
    parser.add_argument(
        "--joint-topic",
        type=str,
        default="/robinion/joint_commands",
        help="Topic for the processed joint position targets.",
    )
    parser.add_argument(
        "--action-topic",
        type=str,
        default="/robinion/policy_actions",
        help="Topic for the raw policy actions.",
    )
    parser.add_argument("--no-actions", action="store_true", default=False, help="Do not publish raw policy actions.")
    parser.add_argument("--no-ros", action="store_true", default=False, help="Run without ROS 2 publishers.")
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)

    if args_cli.ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args_cli.ros_domain_id)
    if os.environ.get("ROS_LOCALHOST_ONLY") == "1":
        print(
            "[WARN] ROS_LOCALHOST_ONLY=1 is set. The subscribing terminal must export "
            "ROS_LOCALHOST_ONLY=1 as well, or the topics will not appear.",
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

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

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

    task_name = args_cli.task.split(":")[-1]
    resume_path = _resolve_checkpoint(args_cli.checkpoint, task_name, agent_cfg, args_cli.agent)
    print(f"[INFO] Task: {task_name}", flush=True)
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}", flush=True)
    print(f"[INFO] Checkpoint: {resume_path}", flush=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_name, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policy = _create_runner(env, agent_cfg, resume_path)

        publisher = None
        if not args_cli.no_ros:
            action_term = env.unwrapped.action_manager.get_term(args_cli.action_term)
            joint_names = list(getattr(action_term.IO_descriptor, "joint_names", []))
            if len(joint_names) != action_term.action_dim:
                raise RuntimeError(
                    f"Action term '{args_cli.action_term}' exposes {len(joint_names)} joint names "
                    f"for {action_term.action_dim} actions; it does not look like a joint action term."
                )
            publisher = RosActionPublisher(
                node_name=args_cli.ros_node,
                joint_topic=args_cli.joint_topic,
                action_topic=None if args_cli.no_actions else args_cli.action_topic,
                joint_names=joint_names,
            )

        try:
            _run_playback(env, policy, publisher, args_cli)
        finally:
            if publisher is not None:
                publisher.close()
            env.close()


if __name__ == "__main__":
    main()
