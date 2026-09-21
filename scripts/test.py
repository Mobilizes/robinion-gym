# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import DCMotorCfg

from isaaclab_assets.robots.humanoid import HUMANOID_CFG

from isaaclab.utils.math import euler_xyz_from_quat

# Joint groups are taken from the robinion2 USD effort/velocity limits. Gains are scaled by
# each group's rated torque so weak joints (head, hip/elbow yaw: 4.1 N.m) do not saturate on
# tiny position errors while strong leg joints (thigh/knee: 19.8 N.m) stay stiff enough to
# support the body. Damping is kept near stiffness/20 for well-damped position control.
EFFORT_LIMIT = {
    ".*thigh.*": 19.8,
    ".*knee.*": 19.8,
    ".*hip_roll.*": 9.9,
    ".*shin.*": 9.9,
    ".*ankle.*": 9.9,
    "torso.*": 10.6,
    ".*shoulder.*": 10.6,
    ".*elbow_pitch.*": 10.6,
    ".*hip_yaw.*": 4.1,
    ".*elbow_yaw.*": 4.1,
    ".*head.*": 4.1,
}
VELOCITY_LIMIT = {
    ".*thigh.*": 4.08,
    ".*knee.*": 4.08,
    ".*hip_roll.*": 4.08,
    ".*shin.*": 4.08,
    ".*ankle.*": 4.08,
    "torso.*": 3.14,
    ".*shoulder.*": 3.14,
    ".*elbow_pitch.*": 3.14,
    ".*hip_yaw.*": 4.82,
    ".*elbow_yaw.*": 4.82,
    ".*head.*": 4.82,
}
STIFFNESS = {
    ".*thigh.*": 100.0,
    ".*knee.*": 100.0,
    ".*hip_roll.*": 60.0,
    ".*shin.*": 60.0,
    ".*ankle.*": 60.0,
    "torso.*": 80.0,
    ".*shoulder.*": 50.0,
    ".*elbow_pitch.*": 50.0,
    ".*hip_yaw.*": 20.0,
    ".*elbow_yaw.*": 20.0,
    ".*head.*": 20.0,
}
DAMPING = {
    ".*thigh.*": 5.0,
    ".*knee.*": 5.0,
    ".*hip_roll.*": 3.0,
    ".*shin.*": 3.0,
    ".*ankle.*": 3.0,
    "torso.*": 4.0,
    ".*shoulder.*": 2.5,
    ".*elbow_pitch.*": 2.5,
    ".*hip_yaw.*": 1.0,
    ".*elbow_yaw.*": 1.0,
    ".*head.*": 1.0,
}


class NewRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/tkuai/isaaclab_projects/robinion_gym/robinion.usd/robinion2/robinion2.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*_shoulder_roll_joint": -1.4,
                ".*shoulder_pitch_joint": 0.4,
                # "left_hip_yaw_joint": -0.5,
            },
        ),
        actuators={
            "body": DCMotorCfg(
                joint_names_expr=".*",
                actuator_effort_limit=EFFORT_LIMIT,
                saturation_effort=EFFORT_LIMIT,
                actuator_velocity_limit=VELOCITY_LIMIT,
                stiffness=STIFFNESS,
                damping=DAMPING,
                armature={".*": 0.01},
            )
        },
    )


import math


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 2.5], [0.0, 0.0, 1.0])
    scene_cfg = NewRobotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    robot: Articulation = scene["robot"]

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(robot.actuators["body"])
    joint_ids, joint_names = robot.find_joints(".*")
    print(joint_ids, joint_names)

    body_ids, body_names = robot.find_bodies(".*foot.*")
    print(body_ids, body_names)
    # print(
    #     "hard limits:",
    #     robot.data.joint_pos_limits[0, joint_ids].tolist(),
    #     "\nsoft limits:",
    #     robot.data.soft_joint_pos_limits[0, joint_ids].tolist(),
    #     "\ndefault/target pos:",
    #     robot.data.default_joint_pos[0, joint_ids].tolist(),
    # )

    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    robot.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    # Simulate physics
    while simulation_app.is_running():
        print(robot.data.root_pos_w.torch[:, 2])
        # quat = robot.data.body_quat_w[:, body_ids]
        # print(euler_xyz_from_quat(quat.reshape(-1, 4))[2])

        # joint_effort = torch.zeros_like(robot.data.joint_pos)
        # joint_effort[:, 2] = 6.0
        # joint_effort[:, 4] = 6.0
        # joint_effort[:, 3] = -6.0
        # joint_effort[:, 5] = -6.0
        # joint_effort[:, 7] = -5.0
        # joint_effort[:, 8] = -5.0
        # joint_effort[:, 3] = -10.0
        # joint_effort[:, 13] = -1.0
        # joint_effort[:, 14] = -1.0

        # robot.set_joint_effort_target(joint_effort)

        scene.write_data_to_sim()

        # perform step
        sim.step()

        sim_time += sim_dt
        scene.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
