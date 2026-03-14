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
# Isaac Sim 5.x uses Python 3.11/3.12; the isaacsim.ros2.bridge extension bundles
# a compatible rclpy. LD_LIBRARY_PATH and RMW_IMPLEMENTATION must be set before
# the extension initialises. Try container paths first (/opt/isaacsim_venv), then ~.
def _find_isaac_bridge_ext() -> str | None:
    # 1) Check venv-style installs (pip-based Isaac Sim)
    for base in (_os.environ.get("ISAAC_SIM_VENV"), "/opt/isaacsim_venv", _os.path.expanduser("~/isaacsim_venv")):
        if not base or not _os.path.isdir(base):
            continue
        for py in ("python3.12", "python3.11"):
            ext = _os.path.join(base, "lib", py, "site-packages", "isaacsim", "exts", "isaacsim.ros2.bridge")
            if _os.path.isdir(ext):
                return ext
    # 2) Check standalone / Docker container install (e.g. nvcr.io/nvidia/isaac-sim)
    for standalone in ("/isaac-sim/exts/isaacsim.ros2.bridge",):
        if _os.path.isdir(standalone):
            return standalone
    return None

_ISAAC_BRIDGE_EXT = _find_isaac_bridge_ext()
if _ISAAC_BRIDGE_EXT:
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

import time as _time  # noqa: E402
import numpy as np  # noqa: E402

print("=" * 50)
print("X3plus Isaac (arm + gripper) ROS2 Control Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

_HEADLESS = _os.environ.get("ISAAC_HEADLESS", "1").strip().lower() not in ("0", "false", "no")
CONFIG = {"headless": _HEADLESS, "width": 1280, "height": 720}
print(f"Headless mode: {_HEADLESS}  (set ISAAC_HEADLESS=0 for GUI)")
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom, Usd  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from omni.kit.app import get_app  # noqa: E402

# Enable ROS2 bridge and Replicator before creating world
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.ros2.bridge")
try:
    enable_extension("omni.replicator.core")
except Exception:
    print("Warning: omni.replicator.core not available — video recording disabled")
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

# Grip servo direction is inverted on the real robot (see opus_joint_config.py).
# The STL meshes match the servo convention, so in Isaac Sim we must mirror
# the inversion: grip_sim = GRIP_LOWER + GRIP_UPPER - grip_urdf.
_GRIP_LOWER = -1.54
_GRIP_UPPER = 0.0

def _invert_grip(urdf_val: float) -> float:
    return _GRIP_LOWER + _GRIP_UPPER - urdf_val   # = -1.54 - urdf_val

def _uninvert_grip(sim_val: float) -> float:
    return _GRIP_LOWER + _GRIP_UPPER - sim_val     # same formula (self-inverse)

# Topic names for JointStateTopicSystem
JOINT_STATES_TOPIC = "/x3plus/joint_states"
JOINT_COMMANDS_TOPIC = "/x3plus/joint_commands"

# ---------------------------------------------------------------------------
# Init positions — loaded from shared YAML (single source of truth)
# ---------------------------------------------------------------------------
_INIT_YAML = Path(_os.environ.get(
    "X3PLUS_INIT_POSITIONS_YAML", "/shared/x3plus_config/init_positions.yaml"))

_FALLBACK_INIT = {
    "arm_joint1": 0.0, "arm_joint2": -1.0, "arm_joint3": -0.8,
    "arm_joint4": -1.3416, "arm_joint5": 0.0, "grip_joint": -0.77,
}

def _load_init_positions() -> dict[str, float]:
    try:
        import yaml
        with open(_INIT_YAML) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items()
                    if k in ARM_GRIP_JOINTS}
    except Exception:
        pass
    return dict(_FALLBACK_INIT)

_INIT_POS = _load_init_positions()
DEFAULT_GRIP_POS = _INIT_POS.get("grip_joint", -0.77)
DEFAULT_ARM_GRIP_POSITIONS = [_INIT_POS.get(j, 0.0) for j in ARM_GRIP_JOINTS]
print(f"Init positions (from {_INIT_YAML}): {DEFAULT_ARM_GRIP_POSITIONS}")


# Single source: docker_all/shared/x3plus_isaac (mounted at /shared in Docker)
X3PLUS_DESCRIPTION_DIR = Path(_os.environ.get("X3PLUS_DESCRIPTION_DIR", "/shared/x3plus_isaac"))


def _get_urdf_path() -> Path:
    return (X3PLUS_DESCRIPTION_DIR / "urdf" / "x3plus_isaac.urdf").resolve()


_BLOCK_MASS = 0.01

# Teleport-grip thresholds (URDF convention: 0 = open, -1.3 = closed)
_GRIP_ATTACH_THRESHOLD = -0.6
_GRIP_DETACH_THRESHOLD = -0.3
_GRIP_PARENT_LINK = "arm_link5"

# Block is parked here when "deleted" (off-screen, below ground)
_BLOCK_PARK_POS = np.array([0.0, 0.0, -1.0])


def _get_link_world_pos(art_prim_path: str, link_name: str):
    """Return world-space position (np.array) of an articulation link via USD."""
    stage = omni.usd.get_context().get_stage()
    link_path = f"{art_prim_path}/{link_name}"
    prim = stage.GetPrimAtPath(link_path)
    if not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = world_tf.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def main() -> None:
    urdf_path = _get_urdf_path()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    meshes_dir = urdf_path.parent.parent / "meshes"
    if not (meshes_dir / "X3plus").exists() and not (meshes_dir / "sensor").exists():
        print(f"Warning: meshes not found under {meshes_dir}")
        print("Populate shared/x3plus_isaac/meshes/ (see docker_all/shared/x3plus_isaac/meshes/README.md).")

    print(f"URDF loaded from: {urdf_path}")
    print(f"Meshes dir:       {meshes_dir}")

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

    _GRIPPER_JOINT_NAMES = {"grip_joint", "rlink_joint2", "rlink_joint3",
                            "llink_joint1", "llink_joint2", "llink_joint3"}
    try:
        for name, joint in robot_model.joints.items():
            if name in _GRIPPER_JOINT_NAMES:
                joint.drive.strength = 50000.0
                joint.drive.damping = 2000.0
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
            for name in ROS2_CONTROL_JOINT_NAMES:
                if name in _INIT_POS:
                    default_positions[name] = _INIT_POS[name]
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

    # ---------------------------------------------------------------------------
    # Recording camera + annotator (Replicator)
    # ---------------------------------------------------------------------------
    _RECORDING_RES = (640, 480)
    _RECORDING_FPS = 10
    recording_annot = None
    try:
        import omni.replicator.core as rep
        rec_cam = rep.create.camera(
            position=(0.35, -0.35, 0.35),
            look_at=(0.18, 0.0, 0.02),
        )
        rec_rp = rep.create.render_product(rec_cam, _RECORDING_RES)
        recording_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        recording_annot.attach([rec_rp])
        for _ in range(5):
            world.step(render=True)
        print(f"Recording camera ready ({_RECORDING_RES[0]}x{_RECORDING_RES[1]} @ {_RECORDING_FPS}fps)")
    except Exception as e:
        print(f"Warning: Recording camera setup failed: {e}. Videos will not be available.")
        recording_annot = None

    # ROS2: publisher, subscriber, and latest command storage
    ros_node = None
    joint_states_pub = None
    latest_joint_commands = None  # (names, positions) or None

    try:
        import rclpy
        from sensor_msgs.msg import JointState
        from rosgraph_msgs.msg import Clock
        from builtin_interfaces.msg import Time as TimeMsg
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from std_srvs.srv import Trigger

        class X3PlusROS2Bridge(Node):
            def __init__(self, sim_world) -> None:
                super().__init__("x3plus_isaac_ros2_bridge")
                self._world = sim_world
                self._latest_commands = None  # (list of names, list of positions) or None
                self._block_prim = None
                self._block_prim_path = None
                self._block_live = False     # True while a block is "in play"
                self._grip_attached = False
                self._last_grip_urdf = 0.0
                self._grip_offset = None

                self.declare_parameter("block_x", 0.2)
                self.declare_parameter("block_y", 0.0)
                self.declare_parameter("block_z", 0.03)
                self.declare_parameter("block_size", 0.03)
                self.declare_parameter("block_qx", 0.0)
                self.declare_parameter("block_qy", 0.0)
                self.declare_parameter("block_qz", 0.0)
                self.declare_parameter("block_qw", 1.0)
                self.declare_parameter("recording_filename", "")

                self._recording = False
                self._recording_proc = None
                self._recording_mode = "ffmpeg"
                self._frame_buffer = []
                self._recording_annot = recording_annot
                self._recording_fps = _RECORDING_FPS
                self._recording_res = _RECORDING_RES
                self._last_frame_time = 0.0
                self._video_dir = "/sim_videos"
                self._video_path = ""

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
                self._clock_pub = self.create_publisher(
                    Clock, "/clock",
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST),
                )
                self.create_service(Trigger, "~/spawn_block", self._spawn_block_cb)
                self.create_service(Trigger, "~/delete_block", self._delete_block_cb)
                self.create_service(Trigger, "~/get_block_pose", self._get_block_pose_cb)
                self.create_service(Trigger, "~/start_recording", self._start_recording_cb)
                self.create_service(Trigger, "~/stop_recording", self._stop_recording_cb)

            def _joint_commands_cb(self, msg: JointState) -> None:
                if msg.name and len(msg.position) >= len(msg.name):
                    self._latest_commands = (list(msg.name), list(msg.position[: len(msg.name)]))

            def get_latest_commands(self) -> tuple[list[str], list[float]] | None:
                return self._latest_commands

            def clear_latest_commands(self) -> None:
                self._latest_commands = None

            def publish_clock(self, sim_time_sec: float) -> None:
                msg = Clock()
                sec = int(sim_time_sec)
                msg.clock = TimeMsg(sec=sec, nanosec=int((sim_time_sec - sec) * 1e9))
                self._clock_pub.publish(msg)

            def publish_joint_state(self, names: list[str], positions: list[float]) -> None:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = names
                msg.position = positions
                self._pub.publish(msg)

            def _ensure_block_exists(self):
                """Create the reusable block prim once, parked off-screen."""
                if self._block_prim is not None:
                    return
                name = "pick_block"
                prim_path = "/World/pick_block"
                block = self._world.scene.add(
                    DynamicCuboid(
                        prim_path=prim_path,
                        name=name,
                        position=_BLOCK_PARK_POS.copy(),
                        scale=np.array([0.03, 0.03, 0.03]),
                        color=np.array([0.9, 0.2, 0.2]),
                        mass=_BLOCK_MASS,
                    )
                )
                self._block_prim = block
                self._block_prim_path = prim_path
                print(f"  Block prim created at {prim_path} (parked)")

            def _spawn_block_cb(self, request, response):
                import json as _json
                bx = self.get_parameter("block_x").value
                by = self.get_parameter("block_y").value
                bz = self.get_parameter("block_z").value
                size = self.get_parameter("block_size").value
                qx = self.get_parameter("block_qx").value
                qy = self.get_parameter("block_qy").value
                qz = self.get_parameter("block_qz").value
                qw = self.get_parameter("block_qw").value
                try:
                    self._ensure_block_exists()
                    target_pos = np.array([bx, by, bz - size / 2.0])
                    target_orient = np.array([qw, qx, qy, qz])
                    self._block_prim.set_world_pose(
                        position=target_pos, orientation=target_orient)
                    self._block_prim.set_linear_velocity(np.zeros(3))
                    self._block_prim.set_angular_velocity(np.zeros(3))
                    self._block_live = True
                    self.get_logger().info(
                        f"Block placed at ({bx:.4f}, {by:.4f}, {bz:.4f})")
                    response.success = True
                    response.message = _json.dumps({
                        "prim_path": self._block_prim_path,
                        "position": [bx, by, float(bz - size / 2.0)],
                    })
                except Exception as e:
                    response.success = False
                    response.message = f"Spawn failed: {e}"
                return response

            def _delete_block_cb(self, request, response):
                self._grip_attached = False
                self._grip_offset = None
                self._block_live = False
                if self._block_prim is not None:
                    try:
                        self._block_prim.set_world_pose(position=_BLOCK_PARK_POS.copy())
                        self._block_prim.set_linear_velocity(np.zeros(3))
                        self._block_prim.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
                response.success = True
                response.message = "Block deleted"
                return response

            def _get_block_pose_cb(self, request, response):
                import json as _json
                if self._block_prim is None or not self._block_live:
                    response.success = False
                    response.message = "No block spawned"
                    return response
                try:
                    pos, orient = self._block_prim.get_world_pose()
                    response.success = True
                    response.message = _json.dumps({
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                        "qw": float(orient[0]),
                        "qx": float(orient[1]),
                        "qy": float(orient[2]),
                        "qz": float(orient[3]),
                    })
                except Exception as e:
                    response.success = False
                    response.message = f"Get pose failed: {e}"
                return response

            def _start_recording_cb(self, request, response):
                import os, subprocess
                if self._recording:
                    response.success = False
                    response.message = "Already recording"
                    return response
                if self._recording_annot is None:
                    response.success = False
                    response.message = "Recording not available (Replicator setup failed)"
                    return response
                filename = self.get_parameter("recording_filename").value
                if not filename:
                    filename = f"recording_{int(_time.time())}.mp4"
                os.makedirs(self._video_dir, exist_ok=True)
                self._video_path = os.path.join(self._video_dir, filename)
                w, h = self._recording_res
                # Try ffmpeg pipe (fastest, produces real MP4)
                try:
                    self._recording_proc = subprocess.Popen(
                        [
                            "ffmpeg", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgba",
                            "-s", f"{w}x{h}", "-r", str(self._recording_fps),
                            "-i", "pipe:0",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "fast",
                            self._video_path,
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._recording_mode = "ffmpeg"
                except FileNotFoundError:
                    self._recording_proc = None
                    self._recording_mode = "frames"
                    self._frame_buffer = []
                self._recording = True
                self._last_frame_time = 0.0
                self.get_logger().info(
                    f"Recording started ({self._recording_mode}): {self._video_path}")
                response.success = True
                response.message = self._video_path
                return response

            def _stop_recording_cb(self, request, response):
                if not self._recording:
                    response.success = True
                    response.message = "Not recording"
                    return response
                self._recording = False
                if self._recording_mode == "ffmpeg" and self._recording_proc is not None:
                    try:
                        self._recording_proc.stdin.close()
                        self._recording_proc.wait(timeout=30)
                    except Exception:
                        try:
                            self._recording_proc.kill()
                        except Exception:
                            pass
                    self._recording_proc = None
                elif self._recording_mode == "frames" and hasattr(self, "_frame_buffer"):
                    self._encode_frames_to_video()
                self.get_logger().info(f"Recording saved: {self._video_path}")
                response.success = True
                response.message = self._video_path
                return response

            def _encode_frames_to_video(self):
                """Encode buffered frames to MP4 using cv2 or imageio."""
                frames = getattr(self, "_frame_buffer", [])
                if not frames:
                    return
                # Try cv2
                try:
                    import cv2
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        self._video_path, fourcc, self._recording_fps, (w, h))
                    for f in frames:
                        rgb = f[:, :, :3] if f.shape[2] == 4 else f
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    writer.release()
                    self._frame_buffer = []
                    return
                except Exception:
                    pass
                # Try imageio
                try:
                    import imageio
                    writer = imageio.get_writer(
                        self._video_path, fps=self._recording_fps)
                    for f in frames:
                        rgb = f[:, :, :3] if f.shape[2] == 4 else f
                        writer.append_data(rgb)
                    writer.close()
                    self._frame_buffer = []
                    return
                except Exception:
                    pass
                self.get_logger().warn("No video encoder available — frames lost")
                self._frame_buffer = []

            def write_recording_frame(self):
                if not self._recording or self._recording_annot is None:
                    return
                now = _time.time()
                if now - self._last_frame_time < 1.0 / self._recording_fps:
                    return
                try:
                    data = self._recording_annot.get_data()
                    if data is None or data.size == 0:
                        return
                    if self._recording_mode == "ffmpeg":
                        if self._recording_proc and self._recording_proc.poll() is None:
                            self._recording_proc.stdin.write(data.tobytes())
                    else:
                        self._frame_buffer.append(data.copy())
                    self._last_frame_time = now
                except Exception:
                    pass

        rclpy.init()
        ros_node = X3PlusROS2Bridge(world)
        print(f"ROS2 bridge: publish {JOINT_STATES_TOPIC}, subscribe {JOINT_COMMANDS_TOPIC}")
        print(f"ROS2 bridge: block management services (spawn_block, delete_block, get_block_pose)")
    except ImportError as e:
        print(f"Warning: rclpy not available ({e}). Run with ROS2 environment for ros2_control.")
        ros_node = None

    def _ros2_physics_callback(dt: float) -> None:
        if art is None or ros_node is None:
            return
        try:
            ros_node.publish_clock(world.current_time)

            name_to_idx = {n: art.get_dof_index(n) for n in ROS2_CONTROL_JOINT_NAMES}
            name_to_idx = {n: i for n, i in name_to_idx.items() if i >= 0}
            if not name_to_idx:
                return

            cmd = ros_node.get_latest_commands()
            grip_pos_for_mimic = _invert_grip(DEFAULT_GRIP_POS)
            if cmd is not None:
                names_in, positions_in = cmd
                indices = []
                values = []
                for n, p in zip(names_in, positions_in):
                    if n in ARM_GRIP_JOINTS and n in name_to_idx:
                        indices.append(name_to_idx[n])
                        val = float(p)
                        if n == "grip_joint":
                            val = _invert_grip(val)
                            grip_pos_for_mimic = val
                        values.append(val)
                if indices and values:
                    targets = np.array(values, dtype=np.float64)
                    idx_arr = np.array(indices, dtype=np.int32)
                    art.apply_action(ArticulationAction(joint_positions=targets, joint_indices=idx_arr))
            else:
                default_vals = list(DEFAULT_ARM_GRIP_POSITIONS)
                default_vals[-1] = _invert_grip(default_vals[-1])
                arm_grip_indices = np.array(
                    [art.get_dof_index(n) for n in ARM_GRIP_JOINTS],
                    dtype=np.int32,
                )
                arm_grip_indices = arm_grip_indices[arm_grip_indices >= 0]
                if len(arm_grip_indices) == 6:
                    art.apply_action(ArticulationAction(
                        joint_positions=np.array(default_vals, dtype=np.float64),
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

            # Publish current state — uninvert grip so the controller sees
            # the original URDF convention.
            names_out = []
            positions_out = []
            grip_urdf_actual = 0.0
            pos = art.get_joint_positions()
            for name in ROS2_CONTROL_JOINT_NAMES:
                idx = name_to_idx.get(name)
                if idx is not None and pos is not None and idx < len(pos):
                    names_out.append(name)
                    val = float(pos[idx])
                    if name == "grip_joint":
                        val = _uninvert_grip(val)
                        grip_urdf_actual = val
                    positions_out.append(val)
            if names_out and positions_out:
                ros_node.publish_joint_state(names_out, positions_out)

            # Teleport-based grip: when gripper closes around a block,
            # record the offset between wrist and block, then move the
            # block rigidly with the wrist every physics step.
            ros_node._last_grip_urdf = grip_urdf_actual
            if ros_node._block_live and ros_node._block_prim is not None:
                if not ros_node._grip_attached:
                    if grip_urdf_actual < _GRIP_ATTACH_THRESHOLD:
                        wrist_pos = _get_link_world_pos(prim_path, _GRIP_PARENT_LINK)
                        if wrist_pos is not None:
                            blk_pos, _ = ros_node._block_prim.get_world_pose()
                            ros_node._grip_offset = np.array(blk_pos, dtype=np.float64) - wrist_pos
                            ros_node._grip_attached = True
                            print(f"  GRIP ATTACHED  wrist={wrist_pos}  block={blk_pos}  offset={ros_node._grip_offset}")
                else:
                    if grip_urdf_actual > _GRIP_DETACH_THRESHOLD:
                        ros_node._grip_attached = False
                        ros_node._grip_offset = None
                        print("  GRIP DETACHED")
                    elif ros_node._grip_offset is not None:
                        wrist_pos = _get_link_world_pos(prim_path, _GRIP_PARENT_LINK)
                        if wrist_pos is not None:
                            new_pos = wrist_pos + ros_node._grip_offset
                            ros_node._block_prim.set_world_pose(position=new_pos)
                            ros_node._block_prim.set_linear_velocity(np.zeros(3))
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
            _is_recording = ros_node is not None and ros_node._recording
            world.step(render=(not _HEADLESS) or _is_recording)
            if _is_recording:
                ros_node.write_recording_frame()
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
