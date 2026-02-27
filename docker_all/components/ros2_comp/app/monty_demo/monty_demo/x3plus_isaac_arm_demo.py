#!/usr/bin/env python3
"""
X3plus in Isaac Sim (arm + gripper) with ros2_control topic interface.

Uses x3plus_isaac URDF. Publishes joint state to /x3plus/joint_states and
subscribes to /x3plus/joint_commands for ros2_control (JointStateTopicSystem).
Run the bringup launch in another terminal to start controller_manager.

  python -m monty_demo.x3plus_isaac_arm_demo
"""

import os as _os
import sys as _sys
from pathlib import Path

# ── rclpy environment setup (MUST happen before SimulationApp) ──────────────
# Isaac Sim uses Python 3.11 but system ROS 2 Jazzy ships rclpy for 3.12.
# The isaacsim.ros2.bridge extension bundles a 3.11-compatible rclpy, but it
# only loads successfully when LD_LIBRARY_PATH and RMW_IMPLEMENTATION are set
# BEFORE the extension initialises.
_ISAAC_BRIDGE_EXT = _os.path.join(
    _os.path.expanduser("~"),
    "isaacsim_venv", "lib", "python3.11", "site-packages",
    "isaacsim", "exts", "isaacsim.ros2.bridge",
)
_ROS_DISTRO = _os.environ.get("ROS_DISTRO", "jazzy")
_INTERNAL_RCLPY_DIR = _os.path.join(_ISAAC_BRIDGE_EXT, _ROS_DISTRO, "rclpy")
_INTERNAL_LIB_DIR = _os.path.join(_ISAAC_BRIDGE_EXT, _ROS_DISTRO, "lib")

if _os.path.isdir(_INTERNAL_RCLPY_DIR):
    for _mod in [k for k in _sys.modules
                 if k == "rclpy" or k.startswith("rclpy.")
                 or k == "rpyutils" or k.startswith("rpyutils.")
                 or k == "sensor_msgs" or k.startswith("sensor_msgs.")]:
        del _sys.modules[_mod]
    if _INTERNAL_RCLPY_DIR not in _sys.path:
        _sys.path.insert(0, _INTERNAL_RCLPY_DIR)
if _os.path.isdir(_INTERNAL_LIB_DIR):
    _ld = _os.environ.get("LD_LIBRARY_PATH", "")
    if _INTERNAL_LIB_DIR not in _ld:
        _os.environ["LD_LIBRARY_PATH"] = _INTERNAL_LIB_DIR + ":" + _ld
if not _os.environ.get("RMW_IMPLEMENTATION"):
    _os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# ────────────────────────────────────────────────────────────────────────────

import numpy as np  # noqa: E402

print("=" * 50)
print("X3plus Isaac (arm + gripper) ROS2 Control Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from omni.kit.app import get_app  # noqa: E402

# Enable ROS2 bridge before creating world
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.ros2.bridge")
ext_manager = get_app().get_extension_manager()
for _ in range(20):
    simulation_app.update()
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        break
print("ROS2 Bridge extension enabled")

# Joint names in ros2_control order (arm + grip + mimic)
ROS2_CONTROL_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5",
    "grip_joint",
    "rlink_joint2", "rlink_joint3", "llink_joint1", "llink_joint2", "llink_joint3",
]
ARM_GRIP_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]
GRIPPER_MIMIC = [
    ("rlink_joint2", -1.0),
    ("rlink_joint3", 1.0),
    ("llink_joint1", -1.0),
    ("llink_joint2", 1.0),
    ("llink_joint3", -1.0),
]

# Topic names for JointStateTopicSystem
JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

# Default "open" gripper pose (from history: GRIP_TARGET = -1.54/2; must hold every frame like old demo)
DEFAULT_GRIP_POS = -0.77  # grip_joint open position
# Default arm pose when no command (old demo set [0,0,0,0,0, GRIP_TARGET] every step)
DEFAULT_ARM_GRIP_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT_GRIP_POS]


def _get_urdf_path() -> Path:
    # Resolve relative to this package (works from source and when installed)
    pkg_dir = Path(__file__).resolve().parent
    return (pkg_dir / "x3plus_isaac" / "urdf" / "x3plus_isaac.urdf").resolve()


def main() -> None:
    urdf_path = _get_urdf_path()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    meshes_dir = urdf_path.parent.parent / "meshes"
    if not (meshes_dir / "X3plus").exists() and not (meshes_dir / "sensor").exists():
        print(f"Warning: meshes not found under {meshes_dir}")
        print("Copy or symlink from x3plus_robot/meshes (see x3plus_isaac/meshes/README.md)")

    print("\nIsaac Sim started. Creating world and importing X3plus (arm + gripper) URDF...")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    import_config = _urdf.ImportConfig()
    import_config.fix_base = True
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.self_collision = False
    try:
        import_config.parse_mimic = False
    except AttributeError:
        try:
            import_config.set_parse_mimic(False)
        except Exception:
            pass
    try:
        import_config.distance_scale = 1.0
    except AttributeError:
        pass

    MIMIC_JOINTS = {"rlink_joint2", "rlink_joint3", "llink_joint1", "llink_joint2", "llink_joint3"}
    try:
        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
    except Exception:
        result = False
        robot_model = None
    if not result or robot_model is None:
        raise RuntimeError("URDFParseFile failed. Check URDF and that meshes exist under x3plus_isaac/meshes/.")

    try:
        for name, joint in robot_model.joints.items():
            if name in MIMIC_JOINTS:
                joint.drive.strength = 0.0
                joint.drive.damping = 0.0
            else:
                joint.drive.strength = 8000.0
                joint.drive.damping = 400.0
    except Exception as e:
        print(f"Warning: joint drive config failed: {e}")

    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    if not result:
        raise RuntimeError("URDFImportRobot failed.")

    print(f"Imported X3plus (arm + gripper) at prim path: {prim_path}")
    world.reset()

    art = None
    try:
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(prim_path)
        art.initialize()
        if art.num_dof and art.num_dof > 0:
            # Default pose: arm at 0, gripper "open" (grip=-0.77, mimics follow) so it looks correct in Isaac
            default_positions = {}
            for name in ROS2_CONTROL_JOINT_NAMES:
                if name == "grip_joint":
                    default_positions[name] = DEFAULT_GRIP_POS
                elif name in (m[0] for m in GRIPPER_MIMIC):
                    mult = next(m for m in GRIPPER_MIMIC if m[0] == name)[1]
                    default_positions[name] = mult * DEFAULT_GRIP_POS
                else:
                    default_positions[name] = 0.0
            default_pose = np.zeros(art.num_dof, dtype=np.float64)
            for name, pos in default_positions.items():
                idx = art.get_dof_index(name)
                if idx >= 0 and idx < art.num_dof:
                    default_pose[idx] = pos
            art.set_joints_default_state(positions=default_pose)
            art.set_joint_positions(default_pose)
    except Exception:
        pass

    # ROS2: publisher, subscriber, and latest command storage
    ros_node = None
    joint_states_pub = None
    latest_joint_commands = None  # (names, positions) or None

    try:
        import rclpy
        from sensor_msgs.msg import JointState
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        class X3PlusROS2Bridge(Node):
            def __init__(self) -> None:
                super().__init__("x3plus_isaac_ros2_bridge")
                self._latest_commands = None  # (list of names, list of positions) or None
                self._sub = self.create_subscription(
                    JointState,
                    JOINT_COMMANDS_TOPIC,
                    self._joint_commands_cb,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST),
                )
                self._pub = self.create_publisher(
                    JointState,
                    JOINT_STATES_TOPIC,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST),
                )

            def _joint_commands_cb(self, msg: JointState) -> None:
                if msg.name and len(msg.position) >= len(msg.name):
                    self._latest_commands = (list(msg.name), list(msg.position[: len(msg.name)]))

            def get_latest_commands(self) -> tuple[list[str], list[float]] | None:
                return self._latest_commands

            def clear_latest_commands(self) -> None:
                self._latest_commands = None

            def publish_joint_state(self, names: list[str], positions: list[float]) -> None:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = names
                msg.position = positions
                self._pub.publish(msg)

        rclpy.init()
        ros_node = X3PlusROS2Bridge()
        print(f"ROS2 bridge: publish {JOINT_STATES_TOPIC}, subscribe {JOINT_COMMANDS_TOPIC}")
    except ImportError as e:
        print(f"Warning: rclpy not available ({e}). Run with ROS2 environment for ros2_control.")
        ros_node = None

    def _ros2_physics_callback(dt: float) -> None:
        if art is None or ros_node is None:
            return
        try:
            name_to_idx = {n: art.get_dof_index(n) for n in ROS2_CONTROL_JOINT_NAMES}
            name_to_idx = {n: i for n, i in name_to_idx.items() if i >= 0}
            if not name_to_idx:
                return

            cmd = ros_node.get_latest_commands()
            grip_pos_for_mimic = DEFAULT_GRIP_POS
            if cmd is not None:
                names_in, positions_in = cmd
                indices = []
                values = []
                for n, p in zip(names_in, positions_in):
                    if n in ARM_GRIP_JOINTS and n in name_to_idx:
                        indices.append(name_to_idx[n])
                        values.append(float(p))
                        if n == "grip_joint":
                            grip_pos_for_mimic = float(p)
                if indices and values:
                    targets = np.array(values, dtype=np.float64)
                    idx_arr = np.array(indices, dtype=np.int32)
                    art.set_joint_position_targets(targets, joint_indices=idx_arr)
            else:
                arm_grip_indices = np.array(
                    [art.get_dof_index(n) for n in ARM_GRIP_JOINTS],
                    dtype=np.int32,
                )
                arm_grip_indices = arm_grip_indices[arm_grip_indices >= 0]
                if len(arm_grip_indices) == 6:
                    art.set_joint_position_targets(
                        np.array(DEFAULT_ARM_GRIP_POSITIONS, dtype=np.float64),
                        joint_indices=arm_grip_indices,
                    )

            mimic_indices = []
            mimic_positions = []
            for name, mult in GRIPPER_MIMIC:
                idx = art.get_dof_index(name)
                if idx >= 0:
                    mimic_indices.append(idx)
                    mimic_positions.append(mult * grip_pos_for_mimic)
            if mimic_indices and mimic_positions:
                art.set_joint_position_targets(
                    np.array(mimic_positions, dtype=np.float64),
                    joint_indices=np.array(mimic_indices, dtype=np.int32),
                )

            # Publish current state (position only)
            names_out = []
            positions_out = []
            pos = art.get_joint_positions()
            for name in ROS2_CONTROL_JOINT_NAMES:
                idx = name_to_idx.get(name)
                if idx is not None and pos is not None and idx < len(pos):
                    names_out.append(name)
                    positions_out.append(float(pos[idx]))
            if names_out and positions_out:
                ros_node.publish_joint_state(names_out, positions_out)
        except Exception as e:
            print(f"Warning: ROS2 physics callback failed: {e}")

    if art is not None and ros_node is not None:
        world.add_physics_callback("x3plus_ros2_bridge", _ros2_physics_callback)

    print("Running simulation. Start bringup in another terminal for ros2_control. Press Ctrl+C to exit.\n")

    try:
        step = 0
        while True:
            if ros_node is not None:
                try:
                    import rclpy
                    rclpy.spin_once(ros_node, timeout_sec=0)
                except ImportError:
                    pass
            world.step(render=True)
            step += 1
            if step % 500 == 0:
                print(f"Step {step} ...")
    except KeyboardInterrupt:
        print("\nStopped by user")

    if ros_node is not None:
        try:
            import rclpy as _rclpy
            ros_node.destroy_node()
            _rclpy.shutdown()
        except Exception:
            pass

    print("=" * 50)
    print("Simulation complete.")
    print("=" * 50)
    simulation_app.close()


if __name__ == "__main__":
    main()
