#!/usr/bin/env python3
"""
X3plus 2-DOF in Isaac Sim. Simplified robot: car base + first two arm joints only.

Run from repo root with Isaac Sim Python. Ensure meshes are in
src/monty_demo/x3plus_isaac_2dof/meshes/ (see x3plus_isaac_2dof/meshes/README.md).

  Container: /isaac-sim/python.sh src/project/demos/x3plus_isaac_2dof_demo.py
  Host (venv): python src/monty_demo/demos/x3plus_isaac_2dof_demo.py
"""

from pathlib import Path

print("=" * 50)
print("X3plus Isaac 2-DOF Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402


def _get_urdf_path() -> Path:
    demo_dir = Path(__file__).resolve().parent
    return (demo_dir.parent / "x3plus_isaac_2dof" / "urdf" / "x3plus_isaac_2dof.urdf").resolve()


def main() -> None:
    urdf_path = _get_urdf_path()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    meshes_dir = urdf_path.parent.parent / "meshes"
    if not (meshes_dir / "X3plus").exists() and not (meshes_dir / "sensor").exists():
        print(f"Warning: meshes not found under {meshes_dir}")
        print("Copy or symlink from x3plus_isaac or x3plus_robot (see x3plus_isaac_2dof/meshes/README.md)")

    print("\nIsaac Sim started. Creating world and importing X3plus 2-DOF URDF...")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    import_config = _urdf.ImportConfig()
    import_config.fix_base = True
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.self_collision = False
    try:
        import_config.distance_scale = 1.0
    except AttributeError:
        pass

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
    )
    if not result:
        raise RuntimeError("URDFParseFile failed. Check URDF and that meshes exist under x3plus_isaac_2dof/meshes/.")

    try:
        for joint in robot_model.joints.values():
            joint.drive.strength = 500.0
            joint.drive.damping = 50.0
    except Exception:
        pass

    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    if not result:
        raise RuntimeError("URDFImportRobot failed.")

    print(f"Imported X3plus 2-DOF at prim path: {prim_path}")
    print("Running simulation. Press Ctrl+C to exit.\n")

    world.reset()

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
