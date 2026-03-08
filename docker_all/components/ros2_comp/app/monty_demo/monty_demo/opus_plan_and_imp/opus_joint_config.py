"""
Shared joint configuration for X3plus: names, servo mapping, rad/deg conversion,
and initial (home) joint positions.
Used by Isaac Sim, real robot bridge, planner, and test scripts.
"""

import math
import os
from pathlib import Path
from typing import Dict, List

import yaml

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
# grip_joint:   URDF [-1.54, 0]    rad -> servo [180, 30] deg => deg = 180 - (rad + 1.54) / 1.54 * 150
#               (inverted: URDF 0 = open → servo 30°, URDF -1.54 = closed → servo 180°)
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

# Grip (inverted servo direction):
#   deg = 180 - (rad + 1.54) * (150/1.54)   =>   rad = (180 - deg) / (150/1.54) - 1.54
GRIP_RAD_TO_DEG_SCALE = 150.0 / 1.54
GRIP_RAD_OFFSET = 1.54
GRIP_SERVO_MAX = 180.0


def rad_to_deg(joint_name: str, rad: float) -> float:
    """Convert URDF position (radians) to servo angle (degrees). Only for actuated joints."""
    if joint_name not in SERVO_MAP:
        return 0.0
    m = SERVO_MAP[joint_name]
    if joint_name == "grip_joint":
        deg = GRIP_SERVO_MAX - (rad + GRIP_RAD_OFFSET) * GRIP_RAD_TO_DEG_SCALE
    else:
        deg = m["direction"] * (rad * m["scale"]) + m["offset_deg"]
    return max(m["min_deg"], min(m["max_deg"], deg))


def deg_to_rad(joint_name: str, deg: float) -> float:
    """Convert servo angle (degrees) to URDF position (radians). Only for actuated joints."""
    if joint_name not in SERVO_MAP:
        return 0.0
    m = SERVO_MAP[joint_name]
    if joint_name == "grip_joint":
        rad = (GRIP_SERVO_MAX - deg) / GRIP_RAD_TO_DEG_SCALE - GRIP_RAD_OFFSET
    else:
        rad = (deg - m["offset_deg"]) / m["scale"]
    return rad


# ---------------------------------------------------------------------------
# Initial (home) joint positions — loaded from shared YAML
# ---------------------------------------------------------------------------

_INIT_YAML_PATH = Path(
    os.environ.get("X3PLUS_INIT_POSITIONS_YAML",
                   "/shared/x3plus_config/init_positions.yaml")
)

_FALLBACK_INIT: Dict[str, float] = {
    "arm_joint1": 0.0,
    "arm_joint2": -1.0,
    "arm_joint3": -0.8,
    "arm_joint4": -1.3416,
    "arm_joint5": 0.0,
    "grip_joint": -0.77,
}


def load_init_positions(yaml_path: Path | None = None) -> Dict[str, float]:
    """Load initial joint positions from the single-source YAML.

    Falls back to built-in defaults if the file is missing (e.g. during
    unit tests or local development outside Docker).
    """
    path = yaml_path or _INIT_YAML_PATH
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if k in ARM_GRIP_JOINTS}
    except (FileNotFoundError, TypeError, ValueError):
        pass
    return dict(_FALLBACK_INIT)


INIT_POSITIONS: Dict[str, float] = load_init_positions()

INIT_ARM_POSITIONS: List[float] = [
    INIT_POSITIONS[j] for j in ARM_GRIP_JOINTS if j != "grip_joint"
]

INIT_GRIP_POSITION: float = INIT_POSITIONS.get("grip_joint", -0.77)

INIT_ARM_GRIP_POSITIONS: List[float] = INIT_ARM_POSITIONS + [INIT_GRIP_POSITION]
