# x3plus_isaac — shared robot description

Meshes from this directory are mounted at `/shared/x3plus_isaac/meshes` in Docker for both **isaac_comp** and **ros2_comp**. The URDF is generated at Docker build time. Override the URDF lookup dir with env var `X3PLUS_DESCRIPTION_DIR`.

## Single source of truth

The **xacro** at `components/ros2_comp/app/monty_demo/x3plus_robot/urdf/x3plus.urdf.xacro` is the single source of truth for all X3plus URDF variants. The Isaac Sim URDF (`x3plus_isaac.urdf`) is **generated automatically** at `docker compose build` time — both `isaac_comp` and `ros2_comp` Dockerfiles run xacro and bake the result into their images. No manual step needed.

For non-Docker / local development, generate it on the host:

```bash
# From docker_all/ (requires xacro / ROS 2):
./scripts/generate_isaac_urdf.sh
```

This produces `urdf/x3plus_isaac.urdf` with relative mesh paths (`../meshes/...`) and no `<ros2_control>` block.

## Meshes (canonical location)

`meshes/` is the **canonical** location for X3plus mesh files. The ros2_comp Dockerfile COPYs meshes from here into the ROS2 package build, so there is only one authoritative copy.

Layout:

- `X3plus/visual/` and `X3plus/collision/` — base_link, arm_link1–5, rlink1–3, llink1–3 (gripper)
- `sensor/visual/` and `sensor/collision/` — camera_link, laser_link, mono_link

To populate from an external source package:

```bash
./scripts/copy_x3plus_from_src.sh [SOURCE_DIR]
```
