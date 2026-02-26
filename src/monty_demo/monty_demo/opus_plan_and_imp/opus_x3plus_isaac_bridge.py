#!/usr/bin/env python3
"""
X3plus in Isaac Sim (arm + gripper) with ros2_control topic interface.

Uses x3plus_isaac URDF. Publishes joint state to /x3plus/joint_states and
subscribes to /x3plus/joint_commands for ros2_control (JointStateTopicSystem).
Uses opus_joint_config for joint names. Run the bringup launch in another
terminal to start controller_manager.

  python -m monty_demo.opus_plan_and_imp.opus_x3plus_isaac_bridge
"""

import os as _os
import sys as _sys
from pathlib import Path

# ── LD_LIBRARY_PATH fix (MUST run before any shared library is loaded) ──────
# glibc caches LD_LIBRARY_PATH at process startup; os.environ changes later
# have no effect on dlopen().  If the system ROS 2 Jazzy env is sourced,
# /opt/ros/jazzy/lib is in LD_LIBRARY_PATH and dlopen will pick up the
# system's librcl_interfaces (built for Python 3.12) instead of Isaac Sim's
# bundled version (Python 3.11), causing an assertion crash.  Fix: re-exec
# once with Isaac Sim's lib dir prepended so the linker sees it first.
_ISAAC_BRIDGE_EXT = _os.path.join(
    _os.path.expanduser("~"),
    "isaacsim_venv", "lib", "python3.11", "site-packages",
    "isaacsim", "exts", "isaacsim.ros2.bridge",
)
_ROS_DISTRO = _os.environ.get("ROS_DISTRO", "jazzy")
_INTERNAL_RCLPY_DIR = _os.path.join(_ISAAC_BRIDGE_EXT, _ROS_DISTRO, "rclpy")
_INTERNAL_LIB_DIR = _os.path.join(_ISAAC_BRIDGE_EXT, _ROS_DISTRO, "lib")

if _os.path.isdir(_INTERNAL_LIB_DIR) and not _os.environ.get("_ISAAC_LD_READY"):
    _ld = _os.environ.get("LD_LIBRARY_PATH", "")
    if not _ld.startswith(_INTERNAL_LIB_DIR):
        _os.environ["LD_LIBRARY_PATH"] = _INTERNAL_LIB_DIR + (":" + _ld if _ld else "")
    _os.environ["_ISAAC_LD_READY"] = "1"
    if not _os.environ.get("RMW_IMPLEMENTATION"):
        _os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    _os.execve(_sys.executable, [_sys.executable] + _sys.argv, _os.environ)

if not _os.environ.get("RMW_IMPLEMENTATION"):
    _os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# ── rclpy environment setup (MUST happen before SimulationApp) ──────────────
# Purge any previously-loaded ROS 2 Python modules so that imports resolve
# from Isaac Sim's bundled rclpy dir (Python 3.11 compatible).
_ROS2_MODULE_PREFIXES = (
    "rclpy", "rpyutils", "sensor_msgs", "rcl_interfaces",
    "builtin_interfaces", "action_msgs", "rosidl_runtime_py",
    "rosidl_parser", "rosidl_adapter", "rosidl_generator_py",
    "std_msgs", "geometry_msgs", "trajectory_msgs",
    "service_msgs", "type_description_interfaces",
    "unique_identifier_msgs", "rosgraph_msgs",
    "ament_index_python",
)
for _mod in list(_sys.modules):
    if any(_mod == _pfx or _mod.startswith(_pfx + ".") for _pfx in _ROS2_MODULE_PREFIXES):
        del _sys.modules[_mod]

if _os.path.isdir(_INTERNAL_RCLPY_DIR) and _INTERNAL_RCLPY_DIR not in _sys.path:
    _sys.path.insert(0, _INTERNAL_RCLPY_DIR)
# ────────────────────────────────────────────────────────────────────────────

# #region agent log
import json as _json, time as _time
_DLOG = "/home/lq/lqtech/dockers/monty_isaac/.cursor/debug-888899.log"
_dlog_counter = [0]
def _dlog(hyp, loc, msg, data=None):
    _dlog_counter[0] += 1
    with open(_DLOG, "a") as f:
        f.write(_json.dumps({"sessionId": "888899", "id": f"log_{_dlog_counter[0]}", "timestamp": int(_time.time()*1000), "location": loc, "message": msg, "data": data or {}, "hypothesisId": hyp}) + "\n")
# #endregion

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
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from omni.kit.app import get_app  # noqa: E402

from monty_demo.opus_plan_and_imp.opus_joint_config import (
    ARM_GRIP_JOINTS,
    JOINT_NAMES,
    MIMIC_MAP,
)

# Enable ROS2 bridge before creating world
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.ros2.bridge")
ext_manager = get_app().get_extension_manager()
for _ in range(20):
    simulation_app.update()
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        break
print("ROS2 Bridge extension enabled")

# Topic names for JointStateTopicSystem
JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

# Default "open" gripper pose
DEFAULT_GRIP_POS = -0.77
DEFAULT_ARM_GRIP_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT_GRIP_POS]

# Mimic list in order for iteration
GRIPPER_MIMIC = [(name, MIMIC_MAP[name][1]) for name in JOINT_NAMES if name in MIMIC_MAP]


def _get_urdf_path() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    return (pkg_dir.parent / "x3plus_isaac" / "urdf" / "x3plus_isaac.urdf").resolve()


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

    MIMIC_JOINTS_SET = set(MIMIC_MAP)
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
            if name in MIMIC_JOINTS_SET:
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
            default_positions = {}
            for name in JOINT_NAMES:
                if name == "grip_joint":
                    default_positions[name] = DEFAULT_GRIP_POS
                elif name in MIMIC_MAP:
                    mult = MIMIC_MAP[name][1]
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

    ros_node = None
    try:
        import rclpy
        from sensor_msgs.msg import JointState
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        # #region agent log
        _dlog("H1", "bridge:rclpy_import", "rclpy imported OK", {"file": rclpy.__file__})
        # #endregion

        class X3PlusROS2Bridge(Node):
            def __init__(self) -> None:
                super().__init__("opus_x3plus_isaac_bridge")
                self._latest_commands = None
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
                    # #region agent log
                    _dlog("H3", "bridge:cmd_cb", "command received", {"names": list(msg.name), "positions": list(msg.position[:len(msg.name)])})
                    # #endregion

            def get_latest_commands(self) -> tuple[list[str], list[float]] | None:
                return self._latest_commands

            def publish_joint_state(self, names: list[str], positions: list[float]) -> None:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = names
                msg.position = positions
                self._pub.publish(msg)

        rclpy.init()
        ros_node = X3PlusROS2Bridge()
        # #region agent log
        _dlog("H2", "bridge:rclpy_init", "rclpy.init + node created OK", {"node_name": ros_node.get_name()})
        # #endregion
        print(f"ROS2 bridge: publish {JOINT_STATES_TOPIC}, subscribe {JOINT_COMMANDS_TOPIC}")
    except ImportError as e:
        # #region agent log
        _dlog("H1", "bridge:rclpy_import_fail", "rclpy import FAILED", {"error": str(e)})
        # #endregion
        print(f"Warning: rclpy not available ({e}). Run with ROS2 environment for ros2_control.")
        ros_node = None
    except Exception as e:
        # #region agent log
        _dlog("H2", "bridge:rclpy_init_fail", "rclpy init FAILED", {"error": str(e), "type": type(e).__name__})
        # #endregion
        print(f"Warning: rclpy init failed ({e}).")
        ros_node = None

    # #region agent log
    _phys_cb_count = [0]
    # #endregion
    def _ros2_physics_callback(dt: float) -> None:
        if art is None or ros_node is None:
            return
        # #region agent log
        _phys_cb_count[0] += 1
        # #endregion
        try:
            name_to_idx = {n: art.get_dof_index(n) for n in JOINT_NAMES}
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
                    art.apply_action(ArticulationAction(
                        joint_positions=np.array(values, dtype=np.float64),
                        joint_indices=np.array(indices, dtype=np.int32),
                    ))
                    # #region agent log
                    if _phys_cb_count[0] % 100 == 1:
                        _dlog("H4", "bridge:apply_cmd", "applied position targets", {"indices": indices, "values": values, "cb_count": _phys_cb_count[0]})
                    # #endregion
            else:
                arm_grip_indices = np.array(
                    [art.get_dof_index(n) for n in ARM_GRIP_JOINTS],
                    dtype=np.int32,
                )
                arm_grip_indices = arm_grip_indices[arm_grip_indices >= 0]
                if len(arm_grip_indices) == 6:
                    art.apply_action(ArticulationAction(
                        joint_positions=np.array(DEFAULT_ARM_GRIP_POSITIONS, dtype=np.float64),
                        joint_indices=arm_grip_indices,
                    ))

            mimic_indices = []
            mimic_positions = []
            for name, mult in GRIPPER_MIMIC:
                idx = art.get_dof_index(name)
                if idx >= 0:
                    mimic_indices.append(idx)
                    mimic_positions.append(mult * grip_pos_for_mimic)
            if mimic_indices and mimic_positions:
                art.apply_action(ArticulationAction(
                    joint_positions=np.array(mimic_positions, dtype=np.float64),
                    joint_indices=np.array(mimic_indices, dtype=np.int32),
                ))

            names_out = []
            positions_out = []
            pos = art.get_joint_positions()
            for name in JOINT_NAMES:
                idx = name_to_idx.get(name)
                if idx is not None and pos is not None and idx < len(pos):
                    names_out.append(name)
                    positions_out.append(float(pos[idx]))
            if names_out and positions_out:
                ros_node.publish_joint_state(names_out, positions_out)
                # #region agent log
                if _phys_cb_count[0] % 200 == 1:
                    _dlog("H5", "bridge:pub_state", "published joint state", {"names": names_out[:3], "positions": [round(p,4) for p in positions_out[:3]], "total": len(names_out), "cb_count": _phys_cb_count[0]})
                # #endregion
        except Exception as e:
            print(f"Warning: ROS2 physics callback failed: {e}")

    if art is not None and ros_node is not None:
        world.add_physics_callback("opus_x3plus_ros2_bridge", _ros2_physics_callback)
        # #region agent log
        _dlog("H4", "bridge:callback_registered", "physics callback registered", {"art_num_dof": art.num_dof})
        # #endregion
    else:
        # #region agent log
        _dlog("H4", "bridge:callback_NOT_registered", "physics callback NOT registered", {"art_is_none": art is None, "ros_node_is_none": ros_node is None})
        # #endregion

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
