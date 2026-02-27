"""
Shared joint configuration for X3plus: names, servo mapping, rad/deg conversion.
Used by both Isaac Sim bridge and real robot bridge (position-only).
"""

import math

# Order must match ros2_control / JointStateTopicSystem: 6 actuated + 5 mimic
JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
    "rlink_joint2",
    "rlink_joint3",
    "llink_joint1",
    "llink_joint2",
    "llink_joint3",
]

# Actuated joints (have physical servos); order maps to servo IDs 1-6
ARM_GRIP_JOINTS = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
]

# Mimic joint -> (parent_joint_name, multiplier)
MIMIC_MAP = {
    "rlink_joint2": ("grip_joint", -1.0),
    "rlink_joint3": ("grip_joint", 1.0),
    "llink_joint1": ("grip_joint", -1.0),
    "llink_joint2": ("grip_joint", 1.0),
    "llink_joint3": ("grip_joint", -1.0),
}

# Per-joint: servo_id (1-6), offset_deg, scale (rad->deg multiplier), direction, min_deg, max_deg
# arm_joint1-4: URDF [-pi/2, pi/2] rad -> servo [0, 180] deg => deg = rad * (180/pi) + 90
# arm_joint5:   URDF [-pi/2, pi]   rad -> servo [0, 270] deg => same formula, servo allows up to 270
# grip_joint:   URDF [-1.54, 0]    rad -> servo [30, 180] deg => deg = (rad + 1.54) / 1.54 * 150 + 30
SERVO_MAP = {
    "arm_joint1": {
        "servo_id": 1,
        "offset_deg": 90.0,
        "scale": 180.0 / math.pi,
        "direction": 1,
        "min_deg": 0.0,
        "max_deg": 180.0,
    },
    "arm_joint2": {
        "servo_id": 2,
        "offset_deg": 90.0,
        "scale": 180.0 / math.pi,
        "direction": 1,
        "min_deg": 0.0,
        "max_deg": 180.0,
    },
    "arm_joint3": {
        "servo_id": 3,
        "offset_deg": 90.0,
        "scale": 180.0 / math.pi,
        "direction": 1,
        "min_deg": 0.0,
        "max_deg": 180.0,
    },
    "arm_joint4": {
        "servo_id": 4,
        "offset_deg": 90.0,
        "scale": 180.0 / math.pi,
        "direction": 1,
        "min_deg": 0.0,
        "max_deg": 180.0,
    },
    "arm_joint5": {
        "servo_id": 5,
        "offset_deg": 90.0,
        "scale": 180.0 / math.pi,
        "direction": 1,
        "min_deg": 0.0,
        "max_deg": 270.0,
    },
    "grip_joint": {
        "servo_id": 6,
        "offset_deg": 30.0,
        "scale": 150.0 / 1.54,  # (180-30)/1.54 for rad range [-1.54, 0]
        "direction": 1,
        "min_deg": 30.0,
        "max_deg": 180.0,
    },
}

# Grip: deg = (rad + 1.54) * (150/1.54) + 30  =>  rad = (deg - 30) / (150/1.54) - 1.54
GRIP_RAD_TO_DEG_SCALE = 150.0 / 1.54
GRIP_RAD_OFFSET = 1.54
GRIP_DEG_OFFSET = 30.0


def rad_to_deg(joint_name: str, rad: float) -> float:
    """Convert URDF position (radians) to servo angle (degrees). Only for actuated joints."""
    if joint_name not in SERVO_MAP:
        return 0.0
    m = SERVO_MAP[joint_name]
    if joint_name == "grip_joint":
        deg = (rad + GRIP_RAD_OFFSET) * GRIP_RAD_TO_DEG_SCALE + GRIP_DEG_OFFSET
    else:
        deg = m["direction"] * (rad * m["scale"]) + m["offset_deg"]
    return max(m["min_deg"], min(m["max_deg"], deg))


def deg_to_rad(joint_name: str, deg: float) -> float:
    """Convert servo angle (degrees) to URDF position (radians). Only for actuated joints."""
    if joint_name not in SERVO_MAP:
        return 0.0
    m = SERVO_MAP[joint_name]
    if joint_name == "grip_joint":
        rad = (deg - GRIP_DEG_OFFSET) / GRIP_RAD_TO_DEG_SCALE - GRIP_RAD_OFFSET
    else:
        rad = (deg - m["offset_deg"]) / m["scale"]
    return rad
