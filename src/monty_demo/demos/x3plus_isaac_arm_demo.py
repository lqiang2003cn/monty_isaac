#!/usr/bin/env python3
"""
X3plus in Isaac Sim (arm + gripper). Uses x3plus_isaac URDF with relative mesh paths.

Run from repo root with Isaac Sim Python. Ensure meshes are in src/monty_demo/x3plus_isaac/meshes/
(see x3plus_isaac/meshes/README.md).

  Container: /isaac-sim/python.sh src/project/demos/x3plus_isaac_arm_demo.py
  Host (venv): python src/monty_demo/demos/x3plus_isaac_arm_demo.py
"""

from pathlib import Path

import numpy as np

print("=" * 50)
print("X3plus Isaac (arm + gripper) Demo - Starting...")
print("=" * 50)

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402


def _get_urdf_path() -> Path:
    demo_dir = Path(__file__).resolve().parent
    return (demo_dir.parent / "x3plus_isaac" / "urdf" / "x3plus_isaac.urdf").resolve()


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
        import_config.parse_mimic = False  # Use software mimic instead
    except AttributeError:
        try:
            import_config.set_parse_mimic(False)
        except Exception:
            pass
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
        raise RuntimeError("URDFParseFile failed. Check URDF and that meshes exist under x3plus_isaac/meshes/.")

    MIMIC_JOINTS = {"rlink_joint2", "rlink_joint3", "llink_joint1", "llink_joint2", "llink_joint3"}
    try:
        for name, joint in robot_model.joints.items():
            if name in MIMIC_JOINTS:
                joint.drive.strength = 0.0  # Kinematically driven by software mimic
                joint.drive.damping = 0.0
            else:
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

    print(f"Imported X3plus (arm + gripper) at prim path: {prim_path}")
    print("Running simulation. Press Ctrl+C to exit.\n")

    world.reset()

    # Software mimic: drive gripper finger joints from grip_joint every physics step
    art = None
    try:
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(prim_path)
        art.initialize()
        if art.num_dof and art.num_dof > 0:
            art.set_joint_positions([0.0] * art.num_dof)
    except Exception:
        pass

    # Gripper mimic mapping: (joint_name, multiplier) relative to grip_joint
    GRIPPER_MIMIC = [
        ("rlink_joint2", -1.0),
        ("rlink_joint3", 1.0),
        ("llink_joint1", -1.0),
        ("llink_joint2", 1.0),
        ("llink_joint3", -1.0),
    ]

    def _apply_software_mimic(dt: float) -> None:
        if art is None:
            return
        try:
            grip_idx = art.get_dof_index("grip_joint")
            positions = art.get_joint_positions()
            if positions is None or grip_idx < 0:
                return
            grip_pos = float(positions[grip_idx])
            mimic_indices = []
            mimic_positions = []
            for name, mult in GRIPPER_MIMIC:
                idx = art.get_dof_index(name)
                if idx >= 0:
                    mimic_indices.append(idx)
                    mimic_positions.append(mult * grip_pos)
            if mimic_indices and mimic_positions:
                art.set_joint_positions(
                    np.array(mimic_positions, dtype=np.float32),
                    joint_indices=np.array(mimic_indices, dtype=np.int32),
                )
        except Exception:
            pass

    if art is not None:
        world.add_physics_callback("gripper_mimic", _apply_software_mimic)

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
