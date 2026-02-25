#!/usr/bin/env python3
"""
X3plus robot in Isaac Sim - Load URDF with resolved mesh paths and run simulation.

Run from repo root with Isaac Sim Python:
  Container: /isaac-sim/python.sh src/project/demos/x3plus_isaac_demo.py
  Host (venv): python src/monty_demo/demos/x3plus_isaac_demo.py

Requires meshes to be available: either copy/link from lqtech_ros2_x3plus into
src/monty_demo/x3plus_robot/meshes/ (same folder structure), or set environment
variable X3PLUS_MESH_ROOT to the root of the package that contains the meshes directory.

RViz vs Isaac Sim display differences (why the gripper can look wrong):
- RViz uses /joint_states and robot_state_publisher: mimic joints are computed from
  the source joint, so the full pose is consistent. Default is usually all joints at 0.
- Isaac Sim runs physics: mimic joints need finite limits (we fix this in the URDF
  rewrite). We also set initial joint positions to 0 after reset so the pose matches
  RViz at startup. Both use Z-up and meters; we set distance_scale=1 explicitly.

Isaac Sim 5.1.0 API.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

print("=" * 50)
print("X3plus Isaac Sim Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

# Imports after SimulationApp
import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

def _get_demo_dir() -> Path:
    """Directory containing this script (demos/)."""
    return Path(__file__).resolve().parent


def _get_x3plus_robot_dir() -> Path:
    """x3plus_robot directory (sibling of demos/ under monty_demo)."""
    return _get_demo_dir().parent / "x3plus_robot"


def _get_urdf_path() -> Path:
    """Path to the source x3plus.urdf."""
    return _get_x3plus_robot_dir() / "urdf" / "x3plus.urdf"


def _get_mesh_root() -> Path | None:
    """
    Resolve mesh root: X3PLUS_MESH_ROOT env, or x3plus_robot if it has meshes/.
    Returns None if not found or meshes/ missing.
    """
    env_root = os.environ.get("X3PLUS_MESH_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.is_dir() and (p / "meshes").is_dir():
            return p
        return p  # still return so we can show a clear error later
    default = _get_x3plus_robot_dir()
    default = default.resolve()
    if default.is_dir() and (default / "meshes").is_dir():
        return default
    return default if default.is_dir() else None


def rewrite_urdf_for_isaac(urdf_path: Path, mesh_root: Path, output_path: Path) -> None:
    """
    Read URDF, replace package://lqtech_ros2_x3plus/ with absolute mesh_root path,
    and fix gripper mimic joints for Isaac Sim. Writes to output_path.

    Isaac Sim's PhysX Mimic API requires follower joints to have finite limits.
    The X3plus URDF uses type="continuous" (no limits) for the five gripper
    mimic joints, so they are converted to revolute with limits derived from
    grip_joint (lower=-1.54, upper=0): multiplier -1 -> [0, 1.54], multiplier 1 -> [-1.54, 0].
    """
    content = urdf_path.read_text(encoding="utf-8")
    mesh_root_abs = mesh_root.resolve().as_posix()
    if not mesh_root_abs.endswith("/"):
        mesh_root_abs += "/"
    content = content.replace("package://lqtech_ros2_x3plus/", mesh_root_abs)

    # Convert continuous mimic joints to revolute with limits so Isaac Sim Mimic API works
    mimic_joints = (
        "rlink_joint2",
        "rlink_joint3",
        "llink_joint1",
        "llink_joint2",
        "llink_joint3",
    )
    for j in mimic_joints:
        content = content.replace(
            f'<joint name="{j}" type="continuous">',
            f'<joint name="{j}" type="revolute">',
        )
    # Add finite limits before each mimic tag (grip_joint range -1.54 to 0)
    content = content.replace(
        '<axis xyz="0 0 1"/>\n        <mimic joint="grip_joint" multiplier="-1"/>',
        '<axis xyz="0 0 1"/>\n        <limit lower="0" upper="1.54" effort="100" velocity="1"/>\n        <mimic joint="grip_joint" multiplier="-1"/>',
    )
    content = content.replace(
        '<axis xyz="0 0 1"/>\n        <mimic joint="grip_joint" multiplier="1"/>',
        '<axis xyz="0 0 1"/>\n        <limit lower="-1.54" upper="0" effort="100" velocity="1"/>\n        <mimic joint="grip_joint" multiplier="1"/>',
    )

    output_path.write_text(content, encoding="utf-8")


def _set_initial_joint_positions(prim_path: str, world: "World") -> None:
    """Set all articulation joint positions to 0 so display matches RViz default pose."""
    try:
        from isaacsim.core.prims import SingleArticulation

        art = SingleArticulation(prim_path)
        art.initialize()
        n = art.num_dof
        if n and n > 0:
            art.set_joint_positions([0.0] * n)
        return
    except Exception:
        pass
    try:
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.articulations import Articulation

        art = Articulation(prim_path)
        art.initialize()
        n = art.num_dof
        if n and n > 0:
            art.set_joint_positions([0.0] * n)
    except Exception:
        pass


def check_mesh_root(mesh_root: Path) -> None:
    """Raise a clear error if mesh root or meshes/ is missing."""
    if not mesh_root.is_dir():
        raise FileNotFoundError(
            f"Mesh root directory not found: {mesh_root}\n"
            "Either copy meshes into src/monty_demo/x3plus_robot/meshes/ (see x3plus_robot/meshes/README.md)\n"
            "or set X3PLUS_MESH_ROOT to the root of the package that contains the 'meshes' directory (e.g. lqtech_ros2_x3plus)."
        )
    meshes_dir = mesh_root / "meshes"
    if not meshes_dir.is_dir():
        raise FileNotFoundError(
            f"meshes/ not found under mesh root: {meshes_dir}\n"
            "Ensure the mesh root directory contains a 'meshes' folder with structure:\n"
            "  meshes/X3plus/visual/, meshes/X3plus/collision/, meshes/sensor/visual/, meshes/sensor/collision/\n"
            "Either copy from lqtech_ros2_x3plus into src/monty_demo/x3plus_robot/meshes/ or set X3PLUS_MESH_ROOT."
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    demo_dir = _get_demo_dir()
    urdf_path = _get_urdf_path()
    mesh_root = _get_mesh_root()

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    if mesh_root is None:
        raise FileNotFoundError(
            "Mesh root could not be resolved.\n"
            "Set X3PLUS_MESH_ROOT to the path that contains the 'meshes' directory,\n"
            "or add meshes under src/monty_demo/x3plus_robot/meshes/ (see x3plus_robot/meshes/README.md)."
        )

    check_mesh_root(mesh_root)

    # Rewrite URDF to a temp file with absolute mesh paths
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        delete=False,
        encoding="utf-8",
    ) as f:
        temp_urdf = Path(f.name)
    try:
        rewrite_urdf_for_isaac(urdf_path, mesh_root, temp_urdf)
        urdf_path_to_use = str(temp_urdf)
    except Exception:
        if temp_urdf.exists():
            temp_urdf.unlink()
        raise

    print("\nIsaac Sim started. Creating world and importing X3plus URDF...")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # URDF import config: fix base, keep mimic joints, no self-collision.
    # distance_scale=1 so URDF units (meters) match Isaac Sim (same as RViz).
    import_config = _urdf.ImportConfig()
    import_config.fix_base = True
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.self_collision = False
    import_config.import_inertia_tensor = True
    import_config.collision_from_visuals = False
    try:
        import_config.parse_mimic = True  # Enable PhysX Mimic API for gripper
    except AttributeError:
        try:
            import_config.set_parse_mimic(True)
        except Exception:
            pass
    try:
        import_config.distance_scale = 1.0
    except AttributeError:
        pass

    # Parse then import so we can tune joint drives if desired
    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=urdf_path_to_use,
        import_config=import_config,
    )
    if not result:
        if temp_urdf.exists():
            temp_urdf.unlink()
        raise RuntimeError("URDFParseFile failed. Check the URDF and mesh paths.")

    # Set joint drive strength/damping (skip mimic joints - they get PhysX Mimic API)
    try:
        for joint_name, joint in robot_model.joints.items():
            mimic = getattr(joint, "mimic", None)
            if mimic and getattr(mimic, "joint", ""):
                continue
            joint.drive.strength = 500.0
            joint.drive.damping = 50.0
    except Exception:
        pass

    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    if temp_urdf.exists():
        temp_urdf.unlink()

    if not result:
        raise RuntimeError("URDFImportRobot failed.")

    print(f"Imported X3plus robot at prim path: {prim_path}")
    print("Running simulation. Press Ctrl+C to exit.\n")

    world.reset()

    # Set initial joint positions to 0 so pose matches RViz default (all joints at 0)
    _set_initial_joint_positions(prim_path, world)

    try:
        step = 0
        while True:
            world.step(render=True)
            step += 1
            if step % 500 == 0:
                print(f"Step {step} ...")
    except KeyboardInterrupt:
        print("\nStopped by user")

    print("=" * 50)
    print("Simulation complete.")
    print("=" * 50)
    simulation_app.close()


if __name__ == "__main__":
    main()
