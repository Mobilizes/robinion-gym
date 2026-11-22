# robinion_gym

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

The registered tasks are:

- `RobinionGym-Velocity-Robinion`
- `RobinionGym-WalkInPlace-Robinion`

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
`uv sync` does not install Isaac Sim. Isaac Lab `main` targets Isaac Sim 6.2 and the CUDA 13 PyTorch stack, which
requires NVIDIA driver `>=595.58.03` on Linux x86_64:

```bash
uv sync
```

This project was generated from an Isaac Lab source checkout. Its `pyproject.toml` uses editable relative paths to that
checkout, so regenerate the project or update `[tool.uv.sources]` if either directory moves.

Optional runtimes are available through extras. Pass the extra to each `uv run` command that needs it. The Robinion
tasks use Newton physics because their native BAM servos cannot run on PhysX:

```bash
# OVRTX renderer with Newton physics
uv run --extra ovrtx isaaclab random_agent --task <TASK_NAME> --viz newton_rtx

# Isaac Sim Kit visualizer with Newton physics
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> --viz kit
```

The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes. Commit `pyproject.toml` and `uv.lock` so collaborators
use the same environment.

## Run the generated tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run python scripts/list_envs.py --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `list_envs.py`.

### Play one environment and publish actions over ROS 2

`scripts/play_ros2.py` runs the `play` workflow with exactly one environment and publishes the commands applied to that
robot:

- `/robinion/joint_commands` (`sensor_msgs/msg/JointState`): processed joint position targets in radians, with joint names.
- `/robinion/policy_actions` (`std_msgs/msg/Float64MultiArray`): raw policy outputs after the RSL-RL action clip.

```bash
# Latest checkpoint
./isaaclab.sh -p scripts/play_ros2.py --checkpoint latest

# Specific checkpoint, paced to the policy step rate
./isaaclab.sh -p scripts/play_ros2.py \
    --checkpoint logs/rsl_rl/robinion_velocity/walk-v5/model_9999.pt --real-time

# Inspect the commands from another terminal
ros2 topic echo /robinion/joint_commands
```

Wait for the `Publishing ... joint targets ...` line in the playback terminal before starting `ros2 topic echo`; the CLI
exits when the topic does not exist yet, and Isaac Sim can take a while to start. If the playback shell sets
`ROS_LOCALHOST_ONLY=1`, export the same value in the echo shell; otherwise FastDDS discovery stays isolated and the
topics do not appear. After changing ROS environment variables, run `ros2 daemon stop` so `ros2 topic list` rebuilds
its cached graph.

The script uses the ROS 2 runtime bundled with Isaac Sim when no sourced Python 3.12 ROS 2 environment is available. On
Ubuntu 22.04 that is the bundled Humble runtime, which interoperates with the system `ros2` CLI. Use
`--ros-distro system` to require an already sourced ROS 2 environment, or `--no-ros` to run playback without publishers.
Use `--ros-domain-id` to override `ROS_DOMAIN_ID`; the subscribing shell must use the same value.

`play_ros2.py` is open-loop from the real robot: the policy runs in Isaac Sim and only its joint targets reach the
hardware. For closed-loop deployment, run the policy on the robot, subscribe to `/robinion/joint_states` and
`/imu/data`, and publish `/robinion/joint_commands`. The sibling `move_actuator` bridge now defaults to
`--rate 200 --state-rate 200`, sets the FTDI USB latency timer to 1 ms, and estimates whether the configured baud rate
can carry one sync read plus one sync write in the 5 ms policy period. See `../move_actuator/README.md` for the
2-3 Mbps bus setup and the timing report.

### Drive one robot with a keyboard or gamepad

`scripts/play_control.py` runs a trained policy in exactly one environment and feeds the velocity command from live
input instead of the random command generator. Command changes are low-pass filtered with a 0.05 s time constant by
default, and the gamepad sticks keep their analog strength, so a partial stick deflection asks for a proportionally
smaller velocity. Releasing an axis stops it immediately instead of coasting on the filter tail. Keyboard state is
tracked from key events and cleared when the Kit window loses focus, so a key released elsewhere cannot leave the
robot walking. Input is not gated on window focus, which keeps keyboard control working through remote-desktop and
livestream clients. Raise `--gamepad-deadzone` if a worn stick does not return to center.

```bash
# Latest checkpoint; the script opens the Kit window because input needs it
./isaaclab.sh -p scripts/play_control.py --checkpoint latest

# Keyboard only, with a stronger push and no strafing
./isaaclab.sh -p scripts/play_control.py --checkpoint latest \
    --input keyboard --max_lin_vel_y 0.0 --push_force 80

# Use a gamepad only; fail fast when none is connected
./isaaclab.sh -p scripts/play_control.py --checkpoint latest --input gamepad
```

Controls:

| Input | Action |
| --- | --- |
| `W` / `S` | forward / backward velocity |
| `A` / `D` | strafe left / right |
| `Q` / `E` | turn left / right |
| `Space` | stop walking and stand still (toggle) |
| `G` | toggle stand / walk mode |
| `F` | shove the robot |
| `R` | reset the robot |
| left stick | forward/back and strafe with analog strength |
| right stick X | turn with analog strength |
| gamepad `B` | stop walking and stand still (toggle) |
| gamepad `Y` | toggle stand / walk mode |
| gamepad `A` | shove the robot |
| gamepad `X` | reset the robot |

The stop/stand toggle (`Space`/gamepad `B`) latches a zero velocity command and selects stand mode, so the robot
stops walking and holds still until you toggle it again. The stand/walk toggle (`G`/gamepad `Y`) switches the
binary gait mode: walk mode follows the velocity command, and walk mode with a zero velocity command walks in
place. Both toggles work in the main velocity task and in the zero-velocity `RobinionGym-WalkInPlace-Robinion`
variant.

Standing is a trained behavior. The velocity task samples a binary `gait_mode` command (`0` = stand, `1` = walk)
for every environment and observes it in the actor and critic. In stand mode the velocity command is forced to
zero and the robot is rewarded for holding the base (`rew_stand_still`) and the init joint pose
(`rew_stand_init_pose`, a heavily weighted L1 kernel), and penalized for airborne feet (`pen_stand_feet_air`).
In walk mode it follows the sampled velocity command; the fraction of walk-mode environments with a zero command
(`rel_standing_envs`) walks in place. One training run therefore covers standing, walking in place, and walking.
A checkpoint trained without these terms may keep stepping at zero command.

The manual shove applies a horizontal force to every body, proportionally to its mass, for `--push-duration` seconds.
The net force defaults to `2.5 * mass` N, and the direction is selected with
`--push-direction {forward,backward,left,right,random}` relative to the robot heading when the push is triggered. Use
`--push-body` with a body-name regex to push only part of the robot, `--push-cooldown` to change the retrigger delay, and
`--no-push-vis` to hide the red force arrow. The trained task's automatic push events are disabled in this script, so the
robot is shoved only when you ask for it.

`--max-lin-vel-x`, `--max-lin-vel-y`, and `--max-ang-vel-z` default to the command ranges the policy was trained on. The
script also sets `--episode-length-s` to effectively infinity by default, so only a fall or `R` resets the robot. Pass a
smaller value to restore periodic time-out resets. The viewport follows the robot from a close third-person view by
default; pass `--no-follow-camera` to free-orbit. Keyboard and gamepad both work at the same time; focus the Kit window
for input to reach the simulation. Pass `--debug-input` to print the keyboard state (window focus, tracked and polled
held bindings), normalized input, target, and filtered command every 0.1 s, which shows whether a key combination
reaches Kit.

Playback skips the task's reward terms because inference does not use them, which keeps the 200 Hz control loop close to
real time. For the crispest response pass `--command-smoothing 0`. If the log warns that playback is running below real
time, raise `--render-interval` to trade render smoothness for wall-clock speed.

### Stand, walk, or walk in place

The main `RobinionGym-Velocity-Robinion` task trains all three behaviors in one run. A binary
`gait_mode` command is added to the command config and to both observation groups: `0` stands
still and `1` walks. In stand mode the velocity command is forced to zero; in walk mode it is
sampled as usual, and walk-mode environments with a zero command walk in place. The gait-shaping
rewards (`feet_gait`, `feet_stance`, `feet_step_length`, `feet_clearance`, `arm_swing_gait`) and the
init-pose reward (`stand_init_pose_l1`) are gated on this bit, while `stand_still_drift` anchors the
base against net drift and `stand_feet_air` makes standing strictly cheaper than stepping in stand
mode. `feet_stance` rewards support-phase contact and `feet_step_length` rewards planting each foot
ahead of the base and past the other foot in the travel direction, which together prevent a
one-leg-leading, one-leg-catching-up gait. Measuring the touchdown against the other foot (rather
than only the base) keeps the requirement symmetric for forward and backward travel.

Train from scratch:

```bash
./isaaclab.sh train --task RobinionGym-Velocity-Robinion --num_envs 4096
```

A checkpoint trained before the gait-mode command has one fewer observation input.
`scripts/pad_checkpoint_obs.py` appends the missing zero column to the actor and critic first
layers (and their optimizer state), so the walking policy can be fine-tuned instead of retrained:

```bash
uv run python scripts/pad_checkpoint_obs.py \
    --checkpoint logs/rsl_rl/robinion_velocity/2026-11-13_19-00-46/model_44996.pt \
    --output logs/rsl_rl/robinion_velocity/walk-v6.1-final-padded.pt \
    --std 0.3

./isaaclab.sh train --task RobinionGym-Velocity-Robinion --num_envs 4096 \
    --checkpoint logs/rsl_rl/robinion_velocity/walk-v6.1-final-padded.pt
```

The converged checkpoint has an exploration std near its floor, so `--std 0.3` reopens exploration;
without it the policy can stay at its old standstill instead of discovering the walk mode. The
padded checkpoint starts at iteration 0, so the full `max_iterations` budget applies to
fine-tuning. From-scratch training is supported too: the push curriculum now holds the shove off
for ~3000 iterations and ramps to the full ±1 m/s range over the next ~12000, so the policy first
learns standing and walking without disturbances. Shorten that ramp only when warm-starting from a
walking checkpoint.

`RobinionGym-WalkInPlace-Robinion` is a zero-velocity specialization of the same task: the
velocity ranges are zero, so walk mode means walking in place and the robot never translates.
Use it only when you want the in-place behavior without translational commands.

Use `play_control.py` with either task to toggle the mode live with `G`/gamepad `Y` or the
stop/stand toggle (`Space`/gamepad `B`); `--start-standing` starts in stand mode instead of walk
mode:

```bash
./isaaclab.sh -p scripts/play_control.py \
    --task RobinionGym-Velocity-Robinion \
    --checkpoint logs/rsl_rl/robinion_velocity/<run>/model_<iteration>.pt
```

The mode and velocity commands are resampled every 10 s by default during training, so the
policy sees mode changes within an episode. Lengthen `resampling_time_range` in
`CommandsCfg.gait_mode` to keep one mode per episode.

`RobinionGym-Velocity-Robinion-PushRecovery` is a fine-tuning variant for push survival. It
trains at the full play command range with a constant ±0.5 m/s shove (above the ±0.4 m/s
deployment target) and gates the stand posture, ankle, foot-slide, and arm shaping terms during
a 1.5 s recovery window. Warm-start it from a walking checkpoint when the deployment shove is
0.4–0.5 m/s:

```bash
./isaaclab.sh train --task RobinionGym-Velocity-Robinion-PushRecovery --num_envs 4096 \
    --checkpoint logs/rsl_rl/robinion_velocity/2026-11-19_17-21-02/model_20000.pt
```

Use `scripts/pad_checkpoint_obs.py --std 0.3` (or set the actor's `distribution.std_param` to
0.3) when the warm-start checkpoint is converged, so the policy can explore a recovery step.

The `2026-11-21_00-47-51_push-recovery-0.5-std0.3` run reached **99.5% walk / 100% stand**
survival at ±0.4 m/s over 30 s episodes with the full play command range. Its best checkpoint is
`model_25000.pt`: the extra 5k iterations reduced walk survival slightly, so prefer the 25k
checkpoint for play. At the manual push default of 0.5 m/s, walk survival is 98.4%; pass
`--push-force 16` to test the 0.4 m/s target.

### Walk in the air with a real IMU in the observation space

`scripts/play_real_imu.py` runs the same trained policy, but spawns the robot at `--height` (1.5 m by default) and pins
its root position in the air. The root orientation follows the physical IMU, so tilting or rotating the board also
tilts and rotates the whole robot. The policy keeps commanding the walking gait, so the legs cycle while the base
rotates. The actor's `projected_gravity` and `base_ang_vel` observations are replaced with values from the physical
SparkFun IMU in the sibling `imu_read/` checkout:

| Observation | Real-IMU source |
| --- | --- |
| `projected_gravity` | Gravity rotated into the IMU body frame; the first valid sample is treated as the neutral pose (`--no-imu-zero` disables that) |
| `base_ang_vel` | Body-frame angular velocity from finite differences of successive IMU orientations (`--no-imu-ang-vel` keeps the simulated zero) |

Tilt or rotate the board and the robot's body follows the same rotation while the policy sees the matching base
observations. With the default startup zeroing the robot starts upright and rotates relative to that pose;
`--no-imu-zero` uses the board's absolute orientation instead. Before the first valid sample, the demo uses the
simulated base observations; after a dropout it holds the last orientation and decays the angular velocity to zero, so
it keeps running until the sensor reconnects.

The demo needs `pyserial` in the project environment and the `imu_read` checkout next to `robinion_gym` (or pointed to
by `IMU_READ_DIR`):

```bash
# Install the serial dependency once
uv pip install --python .venv/bin/python pyserial

# Latest fully trained run, robot locked at 1.5 m, IMU auto-detected
./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1

# Explicit port and a faster walking command
./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1 \
    --imu-port /dev/ttyUSB0 --command-lin-x 0.6

# Headless smoke test
HEADLESS=1 ./isaaclab.sh -p scripts/play_real_imu.py --checkpoint walk-v6.1 \
    --max_steps 200 --imu-wait 3
```

Useful options: `--height` moves the lock point, `--no-base-follow-imu` keeps the body upright while still injecting
the IMU observations, `--imu-ang-vel-filter` and `--imu-ang-vel-hold` tune the reconstructed gyro, `--print-interval`
controls the IMU/robot telemetry (it prints the injected observation values), and `--imu-wait` controls how long
startup waits for the first valid sample. The script disables the self-collision terminations so a leg graze cannot
reset the air-walking demo; `term_height` stays active as a safety net.

## Project structure

Task families live in `src/robinion_gym/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/robinion`. Add another robot by
creating a sibling of `config/robinion`; add another task by creating a sibling task-family directory.

## Development

Run the generated registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and generates a git-ignored `pyrightconfig.json`. The generated
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

### Isaac Sim / Kit startup fails with a `pxr` (OpenUSD) error

Running the `isaacsim` backend directly can fail during Kit startup with:

```text
RuntimeError: extension class wrapper for base class pxrInternal_v0_25_11__pxrReserved__::TfNotice has not been created yet
...
TypeError: No to_python (by-value) converter found for C++ type: std::vector<pxrInternal_v0_25_11__pxrReserved__::SdfPath, ...>
```

Isaac Sim and the pinned `usd-exchange` package can both ship a namespace `pxr` package, so Python may merge them and
load a mix of two OpenUSD runtimes. Use the `isaaclab.sh` wrapper, which puts Isaac Sim's bundled `pxr` ahead of the
standalone one:

```bash
./isaaclab.sh random_agent --task RobinionGym-Velocity-Robinion --viz kit
```

The wrapper is a local workaround for this repository's environment. Remove it once the pinned USD provider matches
Isaac Sim's bundled OpenUSD version.

### Editor cannot resolve modules

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
