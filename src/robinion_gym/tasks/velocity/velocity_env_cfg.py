# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    projected_gravity,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from robinion_gym.robots import ROBINION_CFG

from . import mdp

##
# Physics presets
##


@configclass
class RobinionPhysicsCfg(PresetCfg):
    """Physics backend presets for the robinion velocity environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    default: PhysxCfg = isaacsim_physx


##
# Scene definition
##


@configclass
class RobinionVelocitySceneCfg(InteractiveSceneCfg):
    """Configuration for the robinion velocity scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = ROBINION_CFG

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!head).*"],
        scale=2.0,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ActorCfg(ObsGroup):
        """Observations for actor group."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5, "offset": [0.0, 0.5]})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            pass

    # observation groups
    actor: ActorCfg = ActorCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.5, "offset": [0.0, 0.5]})
        last_action = ObsTerm(func=mdp.last_action)

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(4.0, 6.0),
    #     params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    # )

    reset_non_finite = EventTerm(
        func=mdp.reset_non_finite_envs,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # override_upper_arm_limits = EventTerm(
    #     func=mdp.override_joint_pos_limits,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=[".*_upper_arm:0", ".*_upper_arm:2"]
    #         ),
    #         "limit": 0.3,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    rew_track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    rew_track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    rew_alive = RewTerm(func=mdp.is_alive, weight=0.15)

    rew_feet_flat_contact = RewTerm(
        func=mdp.feet_flat_contact,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "period": 0.5,
            "offset": [0.0, 0.5],
            "swing_center": 0.25,
            "swing_period": 0.4,
            "force_threshold": 5.0,
        },
    )

    # rew_feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.25,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
    #         "threshold": 0.4,
    #     },
    # )

    rew_feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=2.0,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5],
            "swing_center": 0.25,
            "swing_period": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "command_name": "base_velocity",
        },
    )

    rew_arm_swing = RewTerm(
        func=mdp.arm_swing_gait,
        weight=1.5,
        params={
            "period": 0.5,
            "offset": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*shoulder_pitch_joint"),
            "amplitude": 0.25,
            "std": 0.3,
            "phase_offset": math.pi,
            "command_name": "base_velocity",
        },
    )

    pen_base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.60},
    )
    pen_lin_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    pen_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    pen_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    pen_arm_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*elbow.*|.*shoulder.*")},
    )
    # pen_foot_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot.*")},
    # )
    # pen_foot_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot.*")},
    # )
    pen_joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_pitch_joint")},
    )
    pen_arm_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*elbow.*|.*shoulder.*")},
    )
    pen_dof_action_limit = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    pen_dof_joint_pos = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    pen_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    pen_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        },
    )
    pen_feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw_diff,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )
    pen_feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )
    pen_extended_contact = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0005,
        params={
            "threshold": 1000,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    term_timeout = DoneTerm(func=mdp.time_out, time_out=True)
    term_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})


##
# Environment configuration
##


@configclass
class RobinionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobinionVelocitySceneCfg = RobinionVelocitySceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = RobinionPhysicsCfg()


@configclass
class RobinionVelocityPlayEnvCfg(RobinionVelocityEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
