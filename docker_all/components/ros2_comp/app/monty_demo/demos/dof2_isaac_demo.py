#!/usr/bin/env python3
"""
2-DOF robot arm in Isaac Sim. Uses dof2_robot (primitive geometry, no mesh files).

Run from repo root with Isaac Sim Python:
  python src/monty_demo/demos/dof2_isaac_demo.py
"""

from pathlib import Path

print("=" * 50)
print("DOF2 Robot Isaac Sim Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402


def _get_urdf_path() -> Path:
    demo_dir = Path(__file__).resolve().parent
    return (demo_dir.parent / "dof2_robot" / "urdf" / "dof2_robot.urdf").resolve()


def main() -> None:
    urdf_path = _get_urdf_path()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print("\nIsaac Sim started. Creating world and importing DOF2 robot...")

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
        raise RuntimeError("URDFParseFile failed.")

    try:
        for joint in robot_model.joints.values():
            joint.drive.strength = 200.0
            joint.drive.damping = 20.0
    except Exception:
        pass

    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    if not result:
        raise RuntimeError("URDFImportRobot failed.")

    print(f"Imported DOF2 robot at prim path: {prim_path}")
    print("Running simulation. Press Ctrl+C to exit.\n")

    world.reset()

    try:
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(prim_path)
        art.initialize()
        if art.num_dof and art.num_dof > 0:
            art.set_joint_positions([0.0] * art.num_dof)
    except Exception:
        pass

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
