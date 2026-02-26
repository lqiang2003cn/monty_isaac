# X3plus meshes for Isaac Sim

The URDF in `../urdf/x3plus.urdf` references meshes via the ROS package URL `package://lqtech_ros2_x3plus/meshes/...`. For Isaac Sim, mesh paths are resolved at runtime.

You can provide meshes in either of these ways:

## Option 1: Copy meshes here (this directory)

Copy or symlink the mesh files from the **lqtech_ros2_x3plus** package so that this directory has the same layout as that package’s `meshes/` folder:

- `X3plus/visual/` – e.g. base_link.STL, arm_link1.STL, …
- `X3plus/collision/` – collision STLs for the same links
- `sensor/visual/` – e.g. camera_link.STL, laser_link.STL, mono_link.STL
- `sensor/collision/` – collision STLs for sensors

Example (if lqtech_ros2_x3plus is in your workspace):

```bash
# From this repo root
SRC=path/to/lqtech_ros2_x3plus/meshes
DST=src/monty_demo/x3plus_robot/meshes
cp -r "$SRC"/* "$DST"/
# or: ln -s "$SRC"/* "$DST"/
```

The x3plus Isaac Sim demo will use `src/monty_demo/x3plus_robot` as the mesh root by default when `meshes/` exists here.

## Option 2: Use another package path (no copy)

If you prefer not to copy files, set the environment variable **X3PLUS_MESH_ROOT** to the root of the package that contains the `meshes` directory (e.g. the full path to `lqtech_ros2_x3plus`):

```bash
export X3PLUS_MESH_ROOT=/path/to/lqtech_ros2_x3plus
python src/monty_demo/demos/x3plus_isaac_demo.py
```

The demo will rewrite `package://lqtech_ros2_x3plus/` in the URDF to that path so Isaac Sim can find the mesh files.
