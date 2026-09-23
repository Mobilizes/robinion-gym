"""Robinion v2 humanoid: mixed Dynamixel X-series servos, parallelogram legs, 12.0 V bus."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

MPC_HUMANOID_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))

##
# Servo peak torque
##
# The URDF groups the 29 joints into four servo ratings by peak torque [N·m]:
#   4.1  -> hip yaw, head, elbow yaw
#   9.9  -> hip roll, shins, ankles, left front thigh
#  10.6  -> torso, shoulders, elbow pitch
#  19.8  -> knees, back thighs, right front thigh (parallelogram pair)
# The converted USD carries the URDF velocity limits, so Isaac Lab reads those automatically and
# they are not repeated here. It does not expose the URDF effort limits (the PhysX drive max force
# is unset, so the joint effort limit resolves to zero), so the effort limits and the DC-motor
# stall torque (`saturation_effort`) are declared here from the URDF.
#
# NOTE: the URDF lists the right front thigh at 19.8 N·m and the left at 9.9 N·m even though the
# two front thighs are the same servo in the parallelogram linkage. The asymmetry is preserved
# here to match the URDF; change it if the URDF is corrected.

_LEG_PEAK_TORQUE = {
    ".*_hip_yaw_joint": 4.1,
    ".*_hip_roll_joint": 9.9,
    "right_front_thigh_pitch_joint": 19.8,
    "left_front_thigh_pitch_joint": 9.9,
    ".*_back_thigh_pitch_joint": 19.8,
    ".*_knee_pitch_joint": 19.8,
    ".*_front_shin_pitch_joint": 9.9,
    ".*_back_shin_pitch_joint": 9.9,
    ".*_ankle_pitch_joint": 9.9,
    ".*_ankle_roll_joint": 9.9,
}

_TORSO_ARM_PEAK_TORQUE = {
    "torso_pitch_joint": 10.6,
    ".*_shoulder_pitch_joint": 10.6,
    ".*_shoulder_roll_joint": 10.6,
    ".*_elbow_yaw_joint": 4.1,
    ".*_elbow_pitch_joint": 10.6,
}

_HEAD_PEAK_TORQUE = {
    "head_yaw_joint": 4.1,
    "head_pitch_joint": 4.1,
}

##
# Position-control gains
##
# The URDF does not publish controller gains, so the stiffness/damping are chosen as follows:
#   * stiffness: reach roughly the servo's peak torque at a 0.1-0.2 rad tracking error, scaled
#     down for the arms and head where accurate positioning matters less than compliance.
#   * damping: near critical (zeta ~ 0.8-1.0) for a representative effective link inertia, with
#     extra damping on the ankle joints to keep foot contacts quiet.
# These are starting points; tune them against a rollout before trusting a policy.

_LEG_STIFFNESS = {
    ".*_hip_yaw_joint": 60.0,
    ".*_hip_roll_joint": 90.0,
    ".*_front_thigh_pitch_joint": 100.0,
    ".*_back_thigh_pitch_joint": 100.0,
    ".*_knee_pitch_joint": 120.0,
    ".*_front_shin_pitch_joint": 80.0,
    ".*_back_shin_pitch_joint": 80.0,
    ".*_ankle_pitch_joint": 60.0,
    ".*_ankle_roll_joint": 50.0,
}

_LEG_DAMPING = {
    ".*_hip_yaw_joint": 1.5,
    ".*_hip_roll_joint": 2.2,
    ".*_front_thigh_pitch_joint": 2.5,
    ".*_back_thigh_pitch_joint": 2.5,
    ".*_knee_pitch_joint": 3.0,
    ".*_front_shin_pitch_joint": 2.0,
    ".*_back_shin_pitch_joint": 2.0,
    ".*_ankle_pitch_joint": 1.5,
    ".*_ankle_roll_joint": 1.2,
}

_TORSO_ARM_STIFFNESS = {
    "torso_pitch_joint": 80.0,
    ".*_shoulder_pitch_joint": 50.0,
    ".*_shoulder_roll_joint": 50.0,
    ".*_elbow_yaw_joint": 25.0,
    ".*_elbow_pitch_joint": 40.0,
}

_TORSO_ARM_DAMPING = {
    "torso_pitch_joint": 2.0,
    ".*_shoulder_pitch_joint": 1.2,
    ".*_shoulder_roll_joint": 1.2,
    ".*_elbow_yaw_joint": 0.6,
    ".*_elbow_pitch_joint": 1.0,
}

_HEAD_STIFFNESS = {
    "head_yaw_joint": 15.0,
    "head_pitch_joint": 15.0,
}

_HEAD_DAMPING = {
    "head_yaw_joint": 0.4,
    "head_pitch_joint": 0.4,
}

# Reflected rotor inertia (J_rotor * gear_ratio^2); rotor inertia is not published, this is an
# estimate used to stabilize the joint-space inertia.
_ARMATURE = 0.02

# Standing crouch of the parallelogram legs. The USD models the linkage as independent
# revolute joints (no closed loop), so the knee/back-thigh/shin joints are actuated together
# with the front thigh rather than left passive.
_LEG_CROUCH = 0.3  # rad

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
            enabled_self_collisions=True,
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
            saturation_effort=_LEG_PEAK_TORQUE,
            actuator_effort_limit=_LEG_PEAK_TORQUE,
            joint_effort_limit=_LEG_PEAK_TORQUE,
            stiffness=_LEG_STIFFNESS,
            damping=_LEG_DAMPING,
            armature=_ARMATURE,
        ),
        "torso_arms": DCMotorCfg(
            joint_names_expr=[
                "torso_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            saturation_effort=_TORSO_ARM_PEAK_TORQUE,
            actuator_effort_limit=_TORSO_ARM_PEAK_TORQUE,
            joint_effort_limit=_TORSO_ARM_PEAK_TORQUE,
            stiffness=_TORSO_ARM_STIFFNESS,
            damping=_TORSO_ARM_DAMPING,
            armature=_ARMATURE,
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            saturation_effort=_HEAD_PEAK_TORQUE,
            actuator_effort_limit=_HEAD_PEAK_TORQUE,
            joint_effort_limit=_HEAD_PEAK_TORQUE,
            stiffness=_HEAD_STIFFNESS,
            damping=_HEAD_DAMPING,
            armature=_ARMATURE,
        ),
    },
)
