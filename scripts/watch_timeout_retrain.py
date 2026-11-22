#!/usr/bin/env python3
"""Watch a running RSL-RL training and retrain with weaker pushes if needed.

The watchdog waits for ``model_<check_iter>.pt`` in the active run directory, reads
``Episode_Termination/term_timeout`` around that iteration, and compares its rolling mean with
``--threshold``.  When the metric is below the threshold, the run is stopped, the push velocity
range in :mod:`robinion_gym.tasks.velocity.velocity_env_cfg` is reduced, and training is resumed
from the checkpoint for ``--retrain-iterations`` additional iterations.

If the metric is at or above the threshold, the running training is left untouched.

Example:
    python scripts/watch_timeout_retrain.py --check-iter 20000 --threshold 0.8 --new-push 0.4
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TIMEOUT_TAG = "Episode_Termination/term_timeout"
TASK = "RobinionGym-Velocity-Robinion"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "src/robinion_gym/tasks/velocity/velocity_env_cfg.py"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs/rsl_rl/robinion_velocity"

# The push declaration is matched with its values so a partially edited file can never be
# rewritten blindly.
PUSH_BLOCK_RE = re.compile(
    r"(_PUSH_VELOCITY_RANGE:\s*dict\[str,\s*tuple\[float,\s*float\]\]\s*=\s*\{\s*"
    r'"x":\s*\((?P<x_low>[-+0-9.eE]+),\s*(?P<x_high>[-+0-9.eE]+)\),\s*'
    r'"y":\s*\((?P<y_low>[-+0-9.eE]+),\s*(?P<y_high>[-+0-9.eE]+)\),\s*\})',
    re.DOTALL,
)

logger = logging.getLogger("watch_timeout_retrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-iter", type=int, default=20000, help="Iteration at which the timeout is checked.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Timeout ratio below which retraining starts.")
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of iterations ending at --check-iter averaged to smooth the timeout metric.",
    )
    parser.add_argument("--new-push", type=float, default=0.4, help="New symmetric push velocity bound (m/s).")
    parser.add_argument(
        "--retrain-iterations",
        type=int,
        default=10000,
        help="Additional iterations after resuming (RSL-RL counts from the resumed iteration).",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Suffix for the retrained run directory.")
    parser.add_argument("--task", type=str, default=TASK, help="Task name to retrain.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Velocity environment config to edit.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root of the RSL-RL log tree.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to monitor (default: auto-detect).")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Seconds between watchdog polls.")
    parser.add_argument("--kill-timeout", type=float, default=180.0, help="Seconds to wait for a graceful shutdown.")
    parser.add_argument("--launch-timeout", type=float, default=120.0, help="Seconds to verify the retrain is alive.")
    parser.add_argument("--dry-run", action="store_true", help="Read the metric and log the decision without acting.")
    return parser.parse_args()


def _read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _process_start_time(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except OSError:
        return 0
    return int(fields[21])


def find_train_process(task: str) -> tuple[int, list[str]] | None:
    """Find the Python process that runs ``isaaclab train`` for ``task``."""
    candidates: list[tuple[int, int, list[str]]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        args = _read_cmdline(pid)
        if not args:
            continue
        # The console script path is present only in the Python child, not in the `uv run` parent.
        if not any(arg.endswith("/isaaclab") for arg in args):
            continue
        if "train" not in args or task not in args:
            continue
        if any("watch_timeout_retrain" in arg for arg in args):
            continue
        candidates.append((pid, _process_start_time(pid), args))
    if not candidates:
        return None
    pid, _, args = max(candidates, key=lambda item: item[1])
    return pid, args


def find_run_dir(log_root: Path, pid: int) -> Path | None:
    """Find the run directory whose event file was written by ``pid``."""
    matches = sorted(log_root.glob(f"*/events.out.tfevents.*.{pid}.*"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1].parent
    # Fallback: newest run directory with an event file.
    candidates = [p.parent for p in log_root.glob("*/events.out.tfevents.*")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "run.json").stat().st_mtime if (p / "run.json").exists() else 0.0)


def process_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return state != "Z"


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A group made only of zombies still answers killpg; inspect the members explicitly.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == pgid and process_alive(pid):
                return True
        except (OSError, ProcessLookupError):
            continue
    return False


def wait_for_checkpoint(path: Path, pid: int, poll_seconds: float, log_every: float = 300.0) -> bool:
    """Wait until ``path`` exists and its size is stable, keeping the user informed."""
    last_log = 0.0
    while True:
        if path.exists():
            size = path.stat().st_size
            time.sleep(2.0)
            if path.stat().st_size == size and size > 1_000_000:
                return True
            logger.info("Checkpoint %s is still being written", path.name)
        if not process_alive(pid):
            logger.error("Training process %d exited before %s appeared", pid, path.name)
            return False
        if time.monotonic() - last_log >= log_every:
            latest = sorted(path.parent.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
            logger.info("Waiting for %s (latest checkpoint: %s)", path.name, latest[-1].name if latest else "none")
            last_log = time.monotonic()
        time.sleep(poll_seconds)


def read_timeout(run_dir: Path, check_iter: int, window: int) -> tuple[float, float, int]:
    """Return (rolling mean, value at ``check_iter``, number of samples)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    deadline = time.monotonic() + 300.0
    while True:
        try:
            accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 2000})
            accumulator.Reload()
            if TIMEOUT_TAG not in accumulator.Tags()["scalars"]:
                raise RuntimeError(f"{TIMEOUT_TAG!r} is not present in {run_dir}")
            events = accumulator.Scalars(TIMEOUT_TAG)
            at_check = [event.value for event in events if event.step == check_iter]
            samples = [event.value for event in events if check_iter - window < event.step <= check_iter]
            latest_step = max((event.step for event in events), default=-1)
        except Exception as error:
            if time.monotonic() >= deadline:
                raise
            # The event file is still being appended to; a truncated trailing record can make a
            # reload fail, and the next attempt almost always succeeds.
            logger.warning("Could not read %s (will retry): %s", TIMEOUT_TAG, error)
            time.sleep(10.0)
            continue
        if samples:
            mean = statistics.fmean(samples)
            exact = at_check[-1] if at_check else samples[-1]
            logger.info(
                "Iteration %d: timeout mean=%.4f over %d iterations, value=%.4f (latest logged step: %d)",
                check_iter,
                mean,
                len(samples),
                exact,
                latest_step,
            )
            return mean, exact, len(samples)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"No timeout samples in ({check_iter - window}, {check_iter}] in {run_dir} "
                f"(latest logged step: {latest_step})"
            )
        logger.info("Timeout for iteration %d not flushed yet (latest step %d); retrying", check_iter, latest_step)
        time.sleep(10.0)


def build_config_edit(config_path: Path, new_push: float) -> tuple[str, float]:
    """Validate the edit and return (edited source, current bound)."""
    source = config_path.read_text()
    matches = list(PUSH_BLOCK_RE.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one _PUSH_VELOCITY_RANGE block in {config_path}, found {len(matches)}")
    match = matches[0]
    current_bound = max(abs(float(match.group("x_low"))), abs(float(match.group("x_high"))))
    replacement = (
        "_PUSH_VELOCITY_RANGE: dict[str, tuple[float, float]] = {\n"
        f'    "x": ({-new_push:.1f}, {new_push:.1f}),\n'
        f'    "y": ({-new_push:.1f}, {new_push:.1f}),\n'
        "}"
    )
    edited = source[: match.start()] + replacement + source[match.end() :]
    # Keep the explanatory comment consistent with the reduced force.  The robot is ~8 kg and the
    # push lasts 0.2 s, so the peak force is roughly m * dv / dt.
    edited = edited.replace(
        "1 m/s over the 0.2 s push is about 40 N",
        f"{new_push:.1f} m/s over the 0.2 s push is about {8.0 * new_push / 0.2:.0f} N",
    )
    compile(edited, str(config_path), "exec")
    return edited, current_bound


def apply_config_edit(config_path: Path, edited: str) -> None:
    temporary = config_path.with_suffix(config_path.suffix + ".watchdog.tmp")
    temporary.write_text(edited)
    os.replace(temporary, config_path)


def stop_training(pid: int, timeout: float) -> bool:
    """Stop the training process group, escalating from SIGINT to SIGTERM and SIGKILL."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        logger.info("Training process %d already exited", pid)
        return True

    logger.info("Sending SIGINT to training process group %d", pgid)
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not group_alive(pgid):
            logger.info("Training process group %d stopped", pgid)
            return True
        time.sleep(2.0)

    logger.warning("Training did not stop after %.0f s; sending SIGTERM", timeout)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not group_alive(pgid):
            return True
        time.sleep(1.0)

    logger.warning("Training still alive; sending SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    time.sleep(2.0)
    return not group_alive(pgid)


def training_display(pid: int) -> str:
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return ":1"
    for item in environment:
        if item.startswith(b"DISPLAY="):
            return item.decode().split("=", 1)[1]
    return ":1"


def launch_retrain(args: argparse.Namespace, checkpoint: Path, display: str) -> subprocess.Popen:
    uv = shutil.which("uv") or "uv"
    run_name = args.run_name or f"push-{args.new_push:.1f}"
    command = [
        uv,
        "run",
        "--extra",
        "isaacsim",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        args.task,
        "--max_iterations",
        str(args.retrain_iterations),
        "--checkpoint",
        str(checkpoint),
        "--run_name",
        run_name,
    ]
    log_dir = PROJECT_ROOT / "logs/watchdog"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"retrain_{stamp}.log"
    environment = os.environ.copy()
    environment["DISPLAY"] = display
    environment["PYTHONUNBUFFERED"] = "1"
    logger.info("Launching retrain: %s", " ".join(command))
    logger.info("Retrain output: %s", log_path)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info("Retrain started with PID %d", process.pid)
    return process


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    project_logs = PROJECT_ROOT / "logs/watchdog"
    project_logs.mkdir(parents=True, exist_ok=True)
    lock_file = (project_logs / "watch_timeout_retrain.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another watchdog is already running (lock: %s)", project_logs / "watch_timeout_retrain.lock")
        return 1

    logger.info("Watchdog started (PID %d, project %s)", os.getpid(), PROJECT_ROOT)
    found = find_train_process(args.task)
    if found is None:
        logger.error("No running 'isaaclab train' process found for task %s", args.task)
        return 1
    pid, command = found
    logger.info("Monitoring training PID %d: %s", pid, " ".join(command))

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = find_run_dir(args.log_root, pid)
    if run_dir is None or not run_dir.is_dir():
        logger.error("Could not find the active run directory under %s", args.log_root)
        return 1
    logger.info("Monitoring run directory %s", run_dir)
    logger.info(
        "Plan: at iteration %d, retrain with push +-%.2f m/s if the mean timeout over %d iterations is below %.2f; "
        "otherwise leave training untouched (%d additional iterations after resuming)",
        args.check_iter,
        args.new_push,
        args.window,
        args.threshold,
        args.retrain_iterations,
    )

    checkpoint = run_dir / f"model_{args.check_iter}.pt"
    logger.info("Waiting for checkpoint %s", checkpoint.name)
    if not wait_for_checkpoint(checkpoint, pid, args.poll_seconds):
        return 1

    # Validate the config edit before stopping anything: a failed edit must not leave the user
    # with a killed run.
    try:
        edited, current_bound = build_config_edit(args.config, args.new_push)
    except Exception:
        logger.exception("Cannot prepare the push velocity edit in %s; leaving the training untouched", args.config)
        return 1
    logger.info("Push velocity will change from +-%.2f m/s to +-%.2f m/s", current_bound, args.new_push)

    try:
        mean, exact, samples = read_timeout(run_dir, args.check_iter, args.window)
    except Exception:
        logger.exception("Cannot read %s from %s; leaving the training untouched", TIMEOUT_TAG, run_dir)
        return 1

    if mean >= args.threshold:
        logger.info(
            "Timeout mean %.4f is not below %.2f; leaving training running and exiting",
            mean,
            args.threshold,
        )
        return 0

    logger.info(
        "Timeout mean %.4f is below %.2f; stopping training to retrain with push +-%.2f m/s",
        mean,
        args.threshold,
        args.new_push,
    )
    if args.dry_run:
        logger.info("Dry run: skipping stop, config edit, and retrain")
        return 0

    display = training_display(pid)
    if not stop_training(pid, args.kill_timeout):
        logger.error("Could not stop training process group; not editing the config or retraining")
        return 1

    # Give the simulator a moment to release its GPU context before starting a new one.
    time.sleep(10.0)

    apply_config_edit(args.config, edited)
    logger.info("Updated %s to push +-%.2f m/s", args.config, args.new_push)

    try:
        process = launch_retrain(args, checkpoint, display)
    except Exception:
        logger.exception("Failed to launch the retrain")
        return 1

    deadline = time.monotonic() + args.launch_timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            logger.error("Retrain exited during startup with code %s; see logs/watchdog/", code)
            return 1
        time.sleep(5.0)
    logger.info(
        "Retrain is running from %s for %d more iterations; watchdog exiting",
        checkpoint.name,
        args.retrain_iterations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
