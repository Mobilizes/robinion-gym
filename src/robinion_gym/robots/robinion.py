"""Robinion v2 humanoid: mixed Dynamixel X-series servos, parallelogram legs, 14.8 V bus."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

HUMANOID_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))

##
# Servo peak torque
##
# The URDF takes every joint's effort limit from robinion_description/config/actuators.yaml, which maps
# each joint to a DYNAMIXEL model and to the number of servos driving it in parallel. The peak torques
# below are that file's `effort` values at 14.8 V, multiplied by the servo count [N·m]:
#   XM430-W350      4.8 -> head, elbow yaw
#   XH540-W270     11.7 -> hip roll, front shins, ankles
#   XM540-W270     12.9 -> torso, hip yaw, shoulders, elbow pitch
#   XH540-W270 x2  23.4 -> front thighs, knees (two servos per joint)
# The two parallel-link bars (back thigh, back shin) are not mapped to a servo in that file and are
# passive on the real robot. The USD closes their parallelogram with loop joints, so they follow the
# driven links and must not be commanded or driven here.
#
# The converted USD carries the URDF velocity limits, so Isaac Lab reads those automatically and
# they are not repeated here. It does not expose the URDF effort limits (the PhysX drive max force
# is unset, so the joint effort limit resolves to zero), so the effort limits and the DC-motor
# stall torque (`saturation_effort`) are declared here from the URDF.

_LEG_PEAK_TORQUE = {
    ".*_hip_yaw_joint": 12.9,
    ".*_hip_roll_joint": 11.7,
    ".*_front_thigh_pitch_joint": 23.4,
    ".*_knee_pitch_joint": 23.4,
    ".*_front_shin_pitch_joint": 11.7,
    ".*_ankle_pitch_joint": 11.7,
    ".*_ankle_roll_joint": 11.7,
}

_TORSO_ARM_PEAK_TORQUE = {
    "torso_pitch_joint": 12.9,
    ".*_shoulder_pitch_joint": 12.9,
    ".*_shoulder_roll_joint": 12.9,
    ".*_elbow_yaw_joint": 4.8,
    ".*_elbow_pitch_joint": 12.9,
}

_HEAD_PEAK_TORQUE = {
    "head_yaw_joint": 4.8,
    "head_pitch_joint": 4.8,
}

##
# Position-control gains
##
# The URDF does not publish controller gains, so the stiffness/damping are chosen as follows:
#   * stiffness: reach roughly the servo's peak torque at a 0.1-0.25 rad tracking error, scaled
#     down for the arms and head where accurate positioning matters less than compliance.
#   * damping: near critical (zeta ~ 0.8-1.0) for a representative effective link inertia, with
#     extra damping on the ankle joints to keep foot contacts quiet.
# These are starting points; tune them against a rollout before trusting a policy.

_LEG_STIFFNESS = {
    ".*_hip_yaw_joint": 60.0,
    ".*_hip_roll_joint": 90.0,
    ".*_front_thigh_pitch_joint": 100.0,
    ".*_knee_pitch_joint": 120.0,
    ".*_front_shin_pitch_joint": 80.0,
    ".*_ankle_pitch_joint": 60.0,
    ".*_ankle_roll_joint": 50.0,
}

_LEG_DAMPING = {
    ".*_hip_yaw_joint": 1.5,
    ".*_hip_roll_joint": 2.2,
    ".*_front_thigh_pitch_joint": 2.5,
    ".*_knee_pitch_joint": 3.0,
    ".*_front_shin_pitch_joint": 2.0,
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

##
# Armature
##
# Reflected rotor inertia (J_rotor * gear_ratio^2) from config/actuators.yaml, which carries the same
# 0.038 kg·m^2 prior for all three servo models. Joints driven by two servos in parallel carry twice
# that. It is added to the joint-space inertia to stabilize the joint velocities.
_SERVO_ARMATURE = 0.038
_DUAL_SERVO_ARMATURE = 2.0 * _SERVO_ARMATURE

_LEG_ARMATURE = {
    ".*_hip_yaw_joint": _SERVO_ARMATURE,
    ".*_hip_roll_joint": _SERVO_ARMATURE,
    ".*_front_thigh_pitch_joint": _DUAL_SERVO_ARMATURE,
    ".*_knee_pitch_joint": _DUAL_SERVO_ARMATURE,
    ".*_front_shin_pitch_joint": _SERVO_ARMATURE,
    ".*_ankle_pitch_joint": _SERVO_ARMATURE,
    ".*_ankle_roll_joint": _SERVO_ARMATURE,
}

_TORSO_ARM_ARMATURE = {
    "torso_pitch_joint": _SERVO_ARMATURE,
    ".*_shoulder_pitch_joint": _SERVO_ARMATURE,
    ".*_shoulder_roll_joint": _SERVO_ARMATURE,
    ".*_elbow_yaw_joint": _SERVO_ARMATURE,
    ".*_elbow_pitch_joint": _SERVO_ARMATURE,
}

_HEAD_ARMATURE = {
    "head_yaw_joint": _SERVO_ARMATURE,
    "head_pitch_joint": _SERVO_ARMATURE,
}

# Standing crouch of the parallelogram legs. The asset closes the linkage with loop joints, so the
# back-thigh/back-shin bars follow the links they are tied to. They are passive: they are not part of
# the action space and carry no drive. Their initial angles still match the front chain so the loop
# starts closed.
_LEG_CROUCH = 0.0  # rad

ROBINION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HUMANOID_ASSETS_DIR}/robonionv2.usd",
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
            ".*_shoulder_pitch_joint": 0.0,
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
                ".*_knee_pitch_joint",
                ".*_front_shin_pitch_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            saturation_effort=_LEG_PEAK_TORQUE,
            actuator_effort_limit=_LEG_PEAK_TORQUE,
            joint_effort_limit=_LEG_PEAK_TORQUE,
            stiffness=_LEG_STIFFNESS,
            damping=_LEG_DAMPING,
            armature=_LEG_ARMATURE,
        ),
        # The parallelogram bars are unactuated on the real robot. The loop joints in the USD hold
        # them on the driven links, so they must not receive a position command or drive effort.
        "passive_bars": ImplicitActuatorCfg(
            joint_names_expr=[".*_back_thigh_pitch_joint", ".*_back_shin_pitch_joint"],
            stiffness=0.0,
            damping=0.0,
            actuator_effort_limit=0.0,
            joint_effort_limit=0.0,
            armature=0.0,
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
            armature=_TORSO_ARM_ARMATURE,
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            saturation_effort=_HEAD_PEAK_TORQUE,
            actuator_effort_limit=_HEAD_PEAK_TORQUE,
            joint_effort_limit=_HEAD_PEAK_TORQUE,
            stiffness=_HEAD_STIFFNESS,
            damping=_HEAD_DAMPING,
            armature=_HEAD_ARMATURE,
        ),
    },
)
