"""Robinion v2 humanoid: 21x Dynamixel MX-106 + 2x AX-12A (head), parallelogram legs, 11.1 V bus."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

MPC_HUMANOID_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))


# Datasheet values at the 11.1 V
MX106_STALL_TORQUE = 8.0  # N·m
MX106_NO_LOAD_SPEED = 4.29  # rad/s (41 rpm)
AX12A_STALL_TORQUE = 1.39  # N·m
AX12A_NO_LOAD_SPEED = 5.72  # rad/s (54.6 rpm)

# Servo pos mode param
# TODO: must do some check again
MX106_STIFFNESS = 40.0  # N·m/rad, default P gain 850, 4096 ticks/rev, PWM full scale 885
MX106_DAMPING = 1.9  # N·m·s/rad
AX12A_STIFFNESS = 8.0  # N·m/rad, default compliance slope 32 over 1024 ticks/300 deg
AX12A_DAMPING = 0.24  # N·m·s/rad

# Reflected rotor inertia (J_rotor * gear_ratio^2); rotor inertia is not published, these are estimates.
MX106_ARMATURE = 0.02
AX12A_ARMATURE = 0.002

# Standing crouch of the parallelogram legs. The USD models the linkage as independent
# revolute joints (no closed loop), so the knee/back-thigh/shin joints are actuated together
# with the front thigh rather than left passive.
_LEG_CROUCH = 0.1  # rad

ROBINION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MPC_HUMANOID_ASSETS_DIR}/robonionv2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.56),
        joint_pos={
            ".*_front_thigh_pitch_joint": -_LEG_CROUCH,
            ".*_back_thigh_pitch_joint": -_LEG_CROUCH,
            ".*_knee_pitch_joint": _LEG_CROUCH,
            ".*_front_shin_pitch_joint": _LEG_CROUCH,
            ".*_back_shin_pitch_joint": _LEG_CROUCH,
            ".*_ankle_pitch_joint": -_LEG_CROUCH,
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            "torso_pitch_joint": 0.0,
            "head_.*_joint": 0.0,
            ".*_shoulder_roll_joint": -1.4,
            ".*_shoulder_pitch_joint": 0.4,
            ".*_elbow_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_front_thigh_pitch_joint",
                ".*_back_thigh_pitch_joint",
                ".*_knee_pitch_joint",
                ".*_front_shin_pitch_joint",
                ".*_back_shin_pitch_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            saturation_effort=MX106_STALL_TORQUE,
            actuator_effort_limit=MX106_STALL_TORQUE,
            actuator_velocity_limit=MX106_NO_LOAD_SPEED,
            joint_effort_limit=MX106_STALL_TORQUE,
            joint_velocity_limit=MX106_NO_LOAD_SPEED,
            stiffness=MX106_STIFFNESS,
            damping=MX106_DAMPING,
            armature=MX106_ARMATURE,
        ),
        "torso_arms": DCMotorCfg(
            joint_names_expr=[
                "torso_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            saturation_effort=MX106_STALL_TORQUE,
            actuator_effort_limit=MX106_STALL_TORQUE,
            actuator_velocity_limit=MX106_NO_LOAD_SPEED,
            joint_effort_limit=MX106_STALL_TORQUE,
            joint_velocity_limit=MX106_NO_LOAD_SPEED,
            stiffness=MX106_STIFFNESS,
            damping=MX106_DAMPING,
            armature=MX106_ARMATURE,
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            saturation_effort=AX12A_STALL_TORQUE,
            actuator_effort_limit=AX12A_STALL_TORQUE,
            actuator_velocity_limit=AX12A_NO_LOAD_SPEED,
            joint_effort_limit=AX12A_STALL_TORQUE,
            joint_velocity_limit=AX12A_NO_LOAD_SPEED,
            stiffness=AX12A_STIFFNESS,
            damping=AX12A_DAMPING,
            armature=AX12A_ARMATURE,
        ),
    },
)
