# x3plus_robot URDF — single source of truth

- **x3plus.urdf.xacro** — The **single source of truth** for the X3plus robot description. All URDF variants are derived from this file.

  **xacro args:**

  | Arg | Default | Description |
  |-----|---------|-------------|
  | `robot_name` | `x3plus` | `<robot name="...">` attribute |
  | `mesh_prefix` | `package://monty_demo/meshes` | Prefix for all mesh filenames |
  | `include_ros2_control` | `true` | Include `<ros2_control>` block |

  **ROS2 bringup** uses defaults (processed by `opus_x3plus_bringup.launch.py`).

  **Isaac Sim URDF** is generated automatically at `docker compose build` time (both `isaac_comp` and `ros2_comp` Dockerfiles). For non-Docker use:

  ```bash
  # From docker_all/:
  ./scripts/generate_isaac_urdf.sh
  ```

- **ros2_control_topic.xacro** — Included by `x3plus.urdf.xacro` for JointStateTopicSystem (Isaac Sim / real robot).

- **x3plus.urdf** — Pre-generated legacy URDF. Not used by any component; kept for offline tools.
