# x3plus_isaac

X3plus robot variant for **Isaac Sim**: full model including **gripper**. Single URDF with mesh paths that work in Isaac (relative paths, no `package://`).

## Contents

- **urdf/x3plus_isaac.urdf** — One URDF for Isaac Sim:
  - Mesh paths: `../meshes/X3plus/...` and `../meshes/sensor/...` (relative to `urdf/`).
  - Includes: base_footprint, base_link, imu_link, camera_link, laser_link, arm_link1–5, **gripper** (rlink1–3, llink1–3, grip_joint), mono_link.
  - Gripper mimic joints are **revolute with finite limits** (not continuous) so Isaac Sim’s PhysX Mimic API works correctly.
  - The demo sets `parse_mimic=True` and skips drive params on mimic joints.
  - No `<ros2_control>`.

## Meshes

Put meshes under **meshes/** so they match the URDF. E.g. copy or symlink from `x3plus_robot/meshes/`. See **meshes/README.md**.

## Run in Isaac Sim

From docker_all, with meshes in place under `monty_demo/x3plus_isaac/meshes/`:

```bash
python -m monty_demo.x3plus_isaac_arm_demo
# or: python components/ros2_comp/app/monty_demo/demos/... (if run from docker_all)
```

The demo loads `urdf/x3plus_isaac.urdf`; mesh paths `../meshes/...` resolve to `monty_demo/x3plus_isaac/meshes/`.
