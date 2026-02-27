# X3plus meshes (ROS2 and Isaac Sim)

In this layout, URDF/xacro reference meshes as `package://monty_demo/meshes/...`. Meshes here are installed to `share/monty_demo/meshes/` and used by ROS2 bringup. The same files are synced to `monty_demo/x3plus_isaac/meshes/` for Isaac Sim (relative paths in `x3plus_isaac.urdf`).

Required layout:

- `X3plus/visual/` – e.g. base_link.STL, arm_link1.STL, …
- `X3plus/collision/` – collision STLs for the same links
- `sensor/visual/` – e.g. camera_link.STL, laser_link.STL, mono_link.STL
- `sensor/collision/` – collision STLs for sensors

## Copying from repo `src/`

From the repo root or `docker_all`:

```bash
./docker_all/scripts/copy_x3plus_from_src.sh
# or with a specific source package:
./docker_all/scripts/copy_x3plus_from_src.sh /path/to/lqtech_ros2_x3plus
```

The script copies `urdf/` and `meshes/` from the source into `x3plus_robot/` and syncs meshes to `monty_demo/x3plus_isaac/meshes/`. If the source uses `package://lqtech_ros2_x3plus/meshes`, run the `sed` command printed by the script to switch to `package://monty_demo/meshes` (already applied in the current tree).
