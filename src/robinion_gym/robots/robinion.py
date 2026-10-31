"""Robinion v2 humanoid: mixed Dynamixel X-series servos, parallelogram legs, 14.8 V bus."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg, BamMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

HUMANOID_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))

##
# Actuator model
##
# The robot uses Isaac Lab's Newton-native BAM servo model. The motor fit below is the
# XH540-W270 ``m6`` identification from the sibling ``move_actuator/bam-xseries`` checkout
# (``params/xh540/m6.json``). It is the only X-series model identified so far, so it currently
# drives every joint, including the XM540-W270 and XM430-W350 joints. Split the groups by servo
# type (and give each group its own ``BamMotorCfg``) once those parameter files are identified.
#
# BAM computes the firmware position loop, motor torque, and gearbox friction internally, so
# stiffness and damping are not configured here. The armature (reflected rotor inertia) is still
# a solver property and is set below, doubled for joints driven by two servos in parallel.
#
# The fit's ``q_offset`` is a bench calibration offset and is not part of the servo model. Its
# ``command_delay`` (2.55 ms) is one physics step at the task's 400 Hz rate, so it maps to a
# one-step command lag. The fit does not identify a firmware current limit, so ``max_current`` is
# zero, which disables the limiter as in BAM's bench model.
#
# Native BAM requires Newton's MJWarp solver and ``SimulationCfg.use_newton_actuators=True``.
# The task's physics preset therefore defaults to Newton; PhysX cannot execute these groups.

_XH540_M6_MOTOR = BamMotorCfg(
    model="m6",
    kt=2.851502487440982,
    resistance=2.2120112622593875,
    # X-series encoder (4096 counts/rev), firmware P-gain divisor 128, and PWM limit 885 counts.
    error_gain=(4096.0 / (2.0 * math.pi)) / (128.0 * 885.0),
    max_pwm=1.0,
    max_current=0.0,
    friction_base=0.1314635138196002,
    friction_viscous=0.06997471563975972,
    friction_stribeck=0.017662514931086275,
    dtheta_stribeck=0.1624445730565236,
    alpha=22.16148451993002,
    load_friction_motor=0.21182592876233328,
    load_friction_external=0.029368279861542307,
    load_friction_motor_stribeck=0.0002505631220559417,
    load_friction_external_stribeck=0.105384339380273,
    load_friction_motor_quad=0.009999999999999998,
    load_friction_external_quad=2.4748954454684742e-05,
)
"""XH540-W270 ``m6`` fit from ``move_actuator/bam-xseries/params/xh540/m6.json``."""


def _parallel_servo_motor(motor: BamMotorCfg, count: int) -> BamMotorCfg:
    """Combine ``count`` identical servos driving one joint.

    Parallel servos share the command and speed, so their motor torques add. Using the same
    ``kt`` with ``R / count`` doubles/quadruples the drive torque and back-EMF damping. The
    Coulomb, Stribeck, and viscous friction budgets also add, while the load-dependent linear
    terms already scale with the combined load and only the quadratic terms need dividing.
    """
    return BamMotorCfg(
        model=motor.model,
        kt=motor.kt,
        resistance=motor.resistance / count,
        error_gain=motor.error_gain,
        max_pwm=motor.max_pwm,
        # The drive current is the sum of the parallel servos' currents.
        max_current=motor.max_current * count,
        friction_base=motor.friction_base * count,
        friction_viscous=motor.friction_viscous * count,
        friction_stribeck=motor.friction_stribeck * count,
        dtheta_stribeck=motor.dtheta_stribeck,
        alpha=motor.alpha,
        load_friction_motor=motor.load_friction_motor,
        load_friction_external=motor.load_friction_external,
        load_friction_motor_stribeck=motor.load_friction_motor_stribeck,
        load_friction_external_stribeck=motor.load_friction_external_stribeck,
        load_friction_motor_quad=motor.load_friction_motor_quad / count,
        load_friction_external_quad=motor.load_friction_external_quad / count,
    )


_XH540_M6_DUAL_MOTOR = _parallel_servo_motor(_XH540_M6_MOTOR, 2)
"""Two XH540-W270 servos driving one joint (front thighs and knees)."""

#: Per-environment command latency and supply-voltage spread used for actuator domain randomization.
#: The identified command delay is 2.55 ms (about one 400 Hz step); 1-3 steps covers the servo and
#: bus jitter seen on hardware. The lag is held for 100 physics steps (0.25 s) with a 50% resample
#: chance, so it changes on the timescale of a gait rather than every physics step.
_BAM_MIN_DELAY = 1
_BAM_MAX_DELAY = 3
_BAM_DELAY_HOLD_PROB = 0.5
_BAM_DELAY_UPDATE_PERIOD = 100
_BAM_VIN_RANGE = (14.0, 15.5)

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
# is unset, so the joint effort limit resolves to zero), so the effort limits are declared here
# from the URDF. BAM clips its computed motor torque to these same limits.

_LEG_SINGLE_SERVO_JOINTS = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_front_shin_pitch_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
# The front thighs and knees are driven by two XH540-W270 servos each. They form their own group
# with the combined motor so the native BAM drive produces twice the torque and damping.
_LEG_DUAL_SERVO_JOINTS = [".*_front_thigh_pitch_joint", ".*_knee_pitch_joint"]

_LEG_PEAK_TORQUE = {
    ".*_hip_yaw_joint": 12.9,
    ".*_hip_roll_joint": 11.7,
    ".*_front_thigh_pitch_joint": 23.4,
    ".*_knee_pitch_joint": 23.4,
    ".*_front_shin_pitch_joint": 11.7,
    ".*_ankle_pitch_joint": 11.7,
    ".*_ankle_roll_joint": 11.7,
}
# Native actuator authoring resolves every key against the group's joints, so split the table.
_LEG_SINGLE_PEAK_TORQUE = {name: _LEG_PEAK_TORQUE[name] for name in _LEG_SINGLE_SERVO_JOINTS}
_LEG_DUAL_PEAK_TORQUE = {name: _LEG_PEAK_TORQUE[name] for name in _LEG_DUAL_SERVO_JOINTS}

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
# Armature
##
# Reflected rotor inertia from the fitted m6 model. The prior in config/actuators.yaml is 0.038
# kg·m^2 for all three servo models; the fit gives 0.03704 kg·m^2 for the XH540-W270. Joints driven
# by two servos in parallel carry twice that. It is added to the joint-space inertia to stabilize
# the joint velocities.
_SERVO_ARMATURE = 0.037040189482324744
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
_LEG_SINGLE_ARMATURE = {name: _LEG_ARMATURE[name] for name in _LEG_SINGLE_SERVO_JOINTS}
_LEG_DUAL_ARMATURE = {name: _LEG_ARMATURE[name] for name in _LEG_DUAL_SERVO_JOINTS}

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
_LEG_CROUCH = 0.3  # rad

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
        "legs": BamActuatorCfg(
            joint_names_expr=_LEG_SINGLE_SERVO_JOINTS,
            motor=_XH540_M6_MOTOR,
            kp_fw=800.0,
            vin=14.8,
            vin_range=_BAM_VIN_RANGE,
            min_delay=_BAM_MIN_DELAY,
            max_delay=_BAM_MAX_DELAY,
            delay_hold_prob=_BAM_DELAY_HOLD_PROB,
            delay_update_period=_BAM_DELAY_UPDATE_PERIOD,
            actuator_effort_limit=_LEG_SINGLE_PEAK_TORQUE,
            joint_effort_limit=_LEG_SINGLE_PEAK_TORQUE,
            armature=_LEG_SINGLE_ARMATURE,
        ),
        "legs_dual_servo": BamActuatorCfg(
            joint_names_expr=_LEG_DUAL_SERVO_JOINTS,
            motor=_XH540_M6_DUAL_MOTOR,
            kp_fw=800.0,
            vin=14.8,
            vin_range=_BAM_VIN_RANGE,
            min_delay=_BAM_MIN_DELAY,
            max_delay=_BAM_MAX_DELAY,
            delay_hold_prob=_BAM_DELAY_HOLD_PROB,
            delay_update_period=_BAM_DELAY_UPDATE_PERIOD,
            actuator_effort_limit=_LEG_DUAL_PEAK_TORQUE,
            joint_effort_limit=_LEG_DUAL_PEAK_TORQUE,
            armature=_LEG_DUAL_ARMATURE,
        ),
        # The parallelogram bars are unactuated on the real robot. The loop joints in the USD hold
        # them on the driven links, so they must not receive a position command or drive effort.
        "passive_bars": ImplicitActuatorCfg(
            joint_names_expr=[".*_back_thigh_pitch_joint", ".*_back_shin_pitch_joint"],
            stiffness=0.0,
            damping=0.0,
            # Zero stiffness/damping leaves these joints unactuated. The effort limits must still be
            # positive: MJWarp rejects a zero-width actuator force range when importing the model.
            actuator_effort_limit=23.4,
            joint_effort_limit=23.4,
            armature=0.0,
        ),
        "torso_arms": BamActuatorCfg(
            joint_names_expr=[
                "torso_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            motor=_XH540_M6_MOTOR,
            kp_fw=800.0,
            vin=14.8,
            vin_range=_BAM_VIN_RANGE,
            min_delay=_BAM_MIN_DELAY,
            max_delay=_BAM_MAX_DELAY,
            delay_hold_prob=_BAM_DELAY_HOLD_PROB,
            delay_update_period=_BAM_DELAY_UPDATE_PERIOD,
            actuator_effort_limit=_TORSO_ARM_PEAK_TORQUE,
            joint_effort_limit=_TORSO_ARM_PEAK_TORQUE,
            armature=_TORSO_ARM_ARMATURE,
        ),
        "head": BamActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            motor=_XH540_M6_MOTOR,
            kp_fw=800.0,
            vin=14.8,
            vin_range=_BAM_VIN_RANGE,
            min_delay=_BAM_MIN_DELAY,
            max_delay=_BAM_MAX_DELAY,
            delay_hold_prob=_BAM_DELAY_HOLD_PROB,
            delay_update_period=_BAM_DELAY_UPDATE_PERIOD,
            actuator_effort_limit=_HEAD_PEAK_TORQUE,
            joint_effort_limit=_HEAD_PEAK_TORQUE,
            armature=_HEAD_ARMATURE,
        ),
    },
)
