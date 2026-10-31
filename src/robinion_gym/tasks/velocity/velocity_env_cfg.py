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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_tasks.utils import PresetCfg

from robinion_gym.robots import ROBINION_CFG

from . import mdp

##
# Physics presets
##


@configclass
class RobinionPhysicsCfg(PresetCfg):
    """Physics backend presets for the robinion velocity environment."""

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=128,
            nconmax=32,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    # BAM actuators only run through the Newton-native path, so Newton is the default backend.
    default: NewtonCfg = newton_mjwarp


##
# Scene definition
##


@configclass
class RobinionVelocitySceneCfg(InteractiveSceneCfg):
    """Configuration for the robinion velocity scene."""

    # Ground plane. MuJoCo combines two contact friction coefficients with max(), so the ground
    # sits at the lower bound of the trained range and the foot shapes (randomized per environment)
    # set the effective friction. Keeping the ground fixed also avoids a shared, non-per-env
    # material on a single plane.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
            ),
        ),
    )

    # robot
    robot: ArticulationCfg = ROBINION_CFG

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "^(?!head)(?!.*_back_thigh_pitch_joint)(?!.*_back_shin_pitch_joint)(?!.*_elbow_.*_joint).*"
        ],
        # Full normalized action range is 1.2 rad. The policy needs this range for
        # balance-recovery steps; the std cap and floor keep exploration from running away,
        # so the clip is a guardrail rather than the primary gait constraint.
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.8), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.8), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ActorCfg(ObsGroup):
        """Observations for actor group."""

        # base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        projected_gravity = ObsTerm(
            func=projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.1, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase, params={"period": 0.4, "offset": [0.0, 0.5]}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True

    # observation groups
    actor: ActorCfg = ActorCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        gait_phase = ObsTerm(
            func=mdp.gait_phase, params={"period": 0.4, "offset": [0.0, 0.5]}
        )
        last_action = ObsTerm(func=mdp.last_action)

    # privileged observations
    critic: CriticCfg = CriticCfg()


# Push strength (base velocity change, m/s) used in training: the curriculum ramps from no push up
# to this range. The event default is kept on the same value so a rollout without the curriculum
# (play mode, tests) never shoves the robot harder than the policy was trained for.
_PUSH_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.4, 0.4),
    "y": (-0.4, 0.4),
}


@configclass
class EventCfg:
    """Configuration for events."""

    # Set the hard knee lower limit to 0 rad at startup; the zero-width range is deterministic.
    knee_pitch_lower_limit = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"]),
            "lower_limit_distribution_params": (0.0, 0.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

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
            "velocity_range": (0.0, 0.0),
        },
    )

    # Foot contact randomization. The ground plane is fixed at 0.3, so the sampled foot friction
    # is the effective foot-ground friction. Torsional/rolling friction model sole spin and roll,
    # and the stiffness/damping pair randomizes the MuJoCo contact solref.
    foot_contact = EventTerm(
        func=mdp.randomize_foot_contact,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "friction_range": (0.3, 1.2),
            "torsional_friction_range": (0.0, 0.1),
            "rolling_friction_range": (0.0, 0.1),
            "solref_time_constant_range": (0.01, 0.04),
            "solref_damping_ratio_range": (0.7, 1.3),
        },
    )

    # Mass and inertia are sampled independently so the policy sees both payload and inertia
    # mismatches. Mass stays within +-15%, inertia within +-20%.
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": False,
        },
    )
    randomize_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "diagonal_only": True,
        },
    )

    # BAM strength/gain/friction randomization. Command latency and supply-voltage spread are
    # configured on the actuator groups in the robot asset (shared drive settings).
    randomize_actuator = EventTerm(
        func=mdp.randomize_bam_actuator,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_scale_range": (0.8, 1.2),
            "max_effort_scale_range": (0.85, 1.15),
            "friction_scale_range": (0.85, 1.15),
        },
    )

    # force-based shove: the term runs every step and schedules its own pushes
    push_robot = EventTerm(
        func=mdp.push_robot_by_force,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": dict(_PUSH_VELOCITY_RANGE),
            "duration_s": 0.2,
            "push_interval_range_s": (4.0, 6.0),
            "debug_vis": True,
            "force_arrow_scale": 0.05,
        },
    )

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
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    rew_track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    rew_alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # Falling ends the episode; without a direct cost the policy can trade a short unstable
    # episode for a higher instantaneous tracking reward. Keep this moderate so standing
    # still is still worse than walking.
    pen_termination = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # rew_stand_still = RewTerm(
    #     func=mdp.stand_still_drift,
    #     weight=0.75,
    #     params={
    #         "command_name": "base_velocity",
    #         "std": 0.05,
    #         "time_constant": 1.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # rew_feet_flat_contact = RewTerm(
    #     func=mdp.feet_flat_contact,
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
    #         "period": 0.4,
    #         "offset": [0.0, 0.5],
    #         "swing_center": 0.25,
    #         "swing_period": 0.4,
    #         "force_threshold": 5.0,
    #     },
    # )

    rew_feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )

    # rew_feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=1.0,
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
            "period": 0.4,
            "offset": [0.0, 0.5],
            "swing_center": 0.25,
            "swing_period": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "command_name": "base_velocity",
        },
    )

    # rew_feet_stance = RewTerm(
    #     func=mdp.feet_stance,
    #     weight=2.0,
    #     params={
    #         "period": 0.4,
    #         "offset": [0.0, 0.5],
    #         "swing_center": 0.25,
    #         "swing_period": 0.4,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
    #         "command_name": "base_velocity",
    #     },
    # )

    rew_feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "min_height": 0.075,
            "std": 0.05,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )

    rew_arm_swing = RewTerm(
        func=mdp.arm_swing_gait,
        weight=1.5,
        params={
            "period": 0.4,
            "offset": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*shoulder_pitch_joint"),
            "amplitude": 0.2,
            "std": 0.1,
            "phase_offset": math.pi,
            "command_name": "base_velocity",
        },
    )

    pen_base_height = RewTerm(
        func=mdp.base_height_l1,
        weight=-10.0,
        params={"target_height": 0.535},
    )
    pen_lin_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)
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
    pen_ankle_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )
    pen_ankle_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )
    # pen_arm_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.003,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*elbow.*|.*shoulder.*")
    #     },
    # )
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
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_pitch_joint")},
    )
    pen_arm_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*elbow.*|.*shoulder.*")
        },
    )
    pen_arm_wide = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*shoulder_roll_joint")
        },
    )
    pen_dof_joint_pos = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # pen_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)
    pen_flat_orientation = RewTerm(func=mdp.base_rp_l1, weight=-10.0)

    # Contact-rich randomization makes sliding and twisted foot placement much cheaper to exploit,
    # so these shaping terms are weighted above the original -1.0.
    pen_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        },
    )
    pen_feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw_diff,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )
    pen_feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*")},
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-2.0,
        params={
            "left_foot_cfg": SceneEntityCfg("robot", body_names="left_foot.*"),
            "right_foot_cfg": SceneEntityCfg("robot", body_names="right_foot.*"),
            "target_distance": 0.12,
            "command_name": "base_velocity",
            "full_gate_vel": 0.3,
        },
    )

    pen_self_collision = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1e6,
        params={"term_keys": ["term_self_collision"]},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    term_timeout = DoneTerm(func=mdp.time_out, time_out=True)
    term_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.3}
    )
    term_self_collision = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*thigh.*",
                    ".*shin.*",
                    ".*knee.*",
                    ".*hip.*",
                    ".*elbow.*",
                    ".*shoulder.*",
                    ".*gripper.*",
                    ".*upper_body.*",
                    ".*lower_body.*",
                    ".*head.*",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Stage the push disturbance: the full shove arrives before the policy can recover from it
    # (every episode then ends at its first push), so start with no shove and ramp linearly to the
    # trained range. The ramp is keyed on environment steps: 48 steps = 1 iteration, so 240000
    # steps is ~5000 iterations to reach the full shove. Progress is logged under
    # "Curriculum/push_velocity_ramp".
    push_velocity_ramp = CurrTerm(
        func=mdp.push_velocity_range_ramp,
        params={
            "event_name": "push_robot",
            "initial_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
            "final_range": dict(_PUSH_VELOCITY_RANGE),
            "start_step": 48000,
            "end_step": 300000,
        },
    )


##
# Environment configuration
##


@configclass
class RobinionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobinionVelocitySceneCfg = RobinionVelocitySceneCfg(
        num_envs=4096, env_spacing=4.0
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
        self.sim.physics = RobinionPhysicsCfg()
        # Native BAM actuator groups replace the USD drives through Newton's actuator path.
        self.sim.use_newton_actuators = True
        # tick the contact sensor with the physics step so it samples at 200 Hz
        self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self) -> None:
        """Play-mode overrides: play with the full velocity command range and the trained push force."""
        super().play_mode()
        # use the full command range instead of the curriculum's initial level
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # skip the push ramp but keep the shove at the strength the curriculum reaches in training;
        # removing the term alone would leave the event at its default range, which the policy has
        # never seen
        push_ramp = self.curriculum.push_velocity_ramp
        if push_ramp is not None:
            self.events.push_robot.params["velocity_range"] = dict(
                push_ramp.params["final_range"]
            )
        self.curriculum.push_velocity_ramp = None
