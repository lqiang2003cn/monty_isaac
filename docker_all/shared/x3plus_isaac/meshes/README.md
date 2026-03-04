# Meshes for x3plus (canonical location)

This is the **single source** for X3plus mesh files used by both Isaac Sim and ROS2.

- Isaac Sim: URDF at `../urdf/x3plus_isaac.urdf` references meshes via relative paths (`../meshes/...`).
- ROS2: The ros2_comp Dockerfile COPYs these meshes into the ROS2 package build at `share/monty_demo/meshes/`.

Expected layout:

- `X3plus/visual/` and `X3plus/collision/` — base_link, arm_link1–5, rlink1–3, llink1–3 (gripper)
- `sensor/visual/` and `sensor/collision/` — camera_link, laser_link, mono_link

To populate from an external source package:

```bash
./scripts/copy_x3plus_from_src.sh [SOURCE_DIR]
```
