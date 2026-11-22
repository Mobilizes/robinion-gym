# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold the Robinion fixed in the air and command its joints interactively.

The robot base is welded to the world at ``--height``, so joint reaction forces cannot move or
drop the robot. In the Kit viewport, the "Robinion Joint Control" window adds one slider per
actuated joint; dragging a slider writes a position target that the BAM servo model tracks.

.. code-block:: bash

    # Usage (Kit viewport)
    ./isaaclab.sh -p scripts/test.py --viz kit

    # Usage (WebRTC livestream)
    ./isaaclab.sh -p scripts/test.py --livestream 2

    # Suspend the robot higher above the ground
    ./isaaclab.sh -p scripts/test.py --viz kit --height 1.5

"""

"""Parse the command-line arguments first."""


import argparse

from isaaclab.app import add_launcher_args, launch_simulation

# create argparser
parser = argparse.ArgumentParser(description="Test joint motion on a fixed-base Robinion.")
parser.add_argument(
    "--height",
    type=float,
    default=1.0,
    help="Height [m] at which the fixed robot base floats above the ground.",
)
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from robinion_gym.robots import ROBINION_CFG

_PASSIVE_BAR_JOINTS = frozenset(
    {
        "left_back_thigh_pitch_joint",
        "left_back_shin_pitch_joint",
        "right_back_thigh_pitch_joint",
        "right_back_shin_pitch_joint",
    }
)
"""Parallelogram bars follow the driven links and carry no servo drive."""

_JOINT_GROUPS = ("Left leg", "Right leg", "Torso & head", "Left arm", "Right arm")


def _make_robot_cfg(height: float) -> ArticulationCfg:
    """Return the robot config with a world-fixed root suspended at *height*."""
    cfg = ROBINION_CFG.copy()
    cfg.init_state.pos = (0.0, 0.0, height)
    # Use the articulation schema fragments so ``fix_root_link`` goes through the backend-aware
    # writer. The legacy merged cfg relocates the articulation root under Newton.
    cfg.spawn.articulation_props = [
        PhysxArticulationCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        NewtonArticulationCfg(self_collision_enabled=True),
    ]
    cfg.spawn.fix_root_link = True
    return cfg


def _group_joint_indices(joint_names: list[str]) -> dict[str, list[int]]:
    """Group actuated joint indices for the control window."""
    groups: dict[str, list[int]] = {name: [] for name in _JOINT_GROUPS}
    for index, name in enumerate(joint_names):
        if name in _PASSIVE_BAR_JOINTS:
            continue
        if name in ("torso_pitch_joint", "head_yaw_joint", "head_pitch_joint"):
            groups["Torso & head"].append(index)
        elif name.startswith(("left_shoulder", "left_elbow")):
            groups["Left arm"].append(index)
        elif name.startswith(("right_shoulder", "right_elbow")):
            groups["Right arm"].append(index)
        elif name.startswith("left_"):
            groups["Left leg"].append(index)
        elif name.startswith("right_"):
            groups["Right leg"].append(index)
        else:
            groups.setdefault("Other", []).append(index)
    return {group: indices for group, indices in groups.items() if indices}


class JointControlWindow:
    """Kit window with one position-target slider per actuated joint."""

    def __init__(self, robot: Articulation, joint_target):
        """Build the slider window.

        Args:
            robot: The fixed-base robot articulation.
            joint_target: Full position-target tensor, shape ``(1, num_joints)``. Slider changes
                are written into this tensor and applied to the actuators by :meth:`write_targets`.
        """
        import omni.ui as ui

        self._robot = robot
        self._target = joint_target
        self._default_target = joint_target.clone()
        self._models: list[tuple[int, object]] = []
        limits = robot.data.joint_pos_limits.torch[0]

        self._window = ui.Window("Robinion Joint Control", width=500, height=760)
        with self._window.frame, ui.VStack(spacing=6):
            ui.Label(
                "Base is fixed in the air. Drag a slider to command a joint position target.",
                height=0,
                word_wrap=True,
            )
            with ui.ScrollingFrame(), ui.VStack(spacing=4):
                for group_name, joint_ids in _group_joint_indices(robot.joint_names).items():
                    with ui.CollapsableFrame(group_name, collapsed=False), ui.VStack(spacing=3):
                        for joint_id in joint_ids:
                            self._build_joint_row(
                                joint_id,
                                float(limits[joint_id, 0]),
                                float(limits[joint_id, 1]),
                            )
            with ui.HStack(spacing=6, height=28):
                ui.Button("Reset to default pose", clicked_fn=self._reset)
                ui.Button("Log joint positions", clicked_fn=self._log_joint_positions)

    def _build_joint_row(self, joint_id: int, lower: float, upper: float) -> None:
        """Add one label, slider, and numeric field row."""
        import omni.ui as ui

        name = self._robot.joint_names[joint_id]
        value = min(max(float(self._target[0, joint_id]), lower), upper)
        with ui.HStack(spacing=6, height=24):
            ui.Label(name.removesuffix("_joint").replace("_", " "), width=190, tooltip=name)
            model = ui.SimpleFloatModel(value)
            ui.FloatSlider(model=model, min=lower, max=upper, step=0.01, precision=3)
            ui.FloatField(model=model, width=90, precision=3)
        model.add_value_changed_fn(lambda changed, index=joint_id: self._on_value_changed(index, changed))
        self._models.append((joint_id, model))

    def _on_value_changed(self, joint_id: int, model) -> None:
        """Copy one slider value into the target buffer."""
        self._target[0, joint_id] = float(model.get_value_as_float())

    def _reset(self) -> None:
        """Restore the default joint pose in the target buffer and the sliders."""
        self._target.copy_(self._default_target)
        for joint_id, model in self._models:
            model.set_value(float(self._default_target[0, joint_id]))

    def _log_joint_positions(self) -> None:
        """Print the measured joint positions to the console."""
        positions = self._robot.data.joint_pos.torch[0]
        for index, name in enumerate(self._robot.joint_names):
            print(f"[INFO] {name}: {positions[index].item():+.3f} rad", flush=True)

    def write_targets(self) -> None:
        """Send the current slider targets to the actuator collection."""
        self._robot.actuators.target_command.set_position_index(value=self._target, full_data=True)


class NewRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = _make_robot_cfg(args_cli.height)


def main():
    """Main function."""

    # Initialize the simulation context. Native BAM actuators require Newton's MJWarp solver.
    # Gravity stays off so the limbs are weightless while the base is held in the air.
    sim_cfg = SimulationCfg(
        dt=0.01,
        gravity=(0.0, 0.0, 0.0),
        use_newton_actuators=True,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(cone="pyramidal", impratio=1.0, integrator="implicitfast")),
    )
    # Launch the simulator runtime that the configuration needs: Kit for the viewport/livestream,
    # or no Isaac Sim at all for a kitless Newton run. The launcher resolves the visualizer into
    # this same sim_cfg, so it must be passed here and then given to SimulationContext.
    with launch_simulation(sim_cfg, args_cli):
        sim = SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 0.0, args_cli.height + 1.5], [0.0, 0.0, args_cli.height])
        scene_cfg = NewRobotSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        robot: Articulation = scene["robot"]

        # Play the simulator
        sim.reset()
        # Now we are ready!
        print(f"[INFO]: Setup complete; robot base fixed at z={args_cli.height:.2f} m", flush=True)
        print(f"[INFO]: Actuated joints: {[n for n in robot.joint_names if n not in _PASSIVE_BAR_JOINTS]}", flush=True)

        # Hold every joint at the default pose until a slider moves it.
        joint_target = robot.data.default_joint_pos.torch.clone()
        robot.actuators.target_command.set_position_index(value=joint_target, full_data=True)

        control = None
        if sim.has_gui:
            try:
                control = JointControlWindow(robot, joint_target)
                print("[INFO]: Use the 'Robinion Joint Control' window to move the joints.", flush=True)
            except Exception as exc:
                print(f"[WARN]: Joint slider window unavailable: {exc}", flush=True)
        else:
            print("[INFO]: No Kit GUI; joint sliders are disabled.", flush=True)

        # Seed the initial joint targets before stepping.
        scene.write_data_to_sim()

        sim_dt = sim.get_physics_dt()
        sim_time = 0.0

        # Simulate physics
        while sim.is_running():
            if control is not None:
                control.write_targets()

            scene.write_data_to_sim()

            # perform step
            sim.step()

            sim_time += sim_dt
            scene.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
