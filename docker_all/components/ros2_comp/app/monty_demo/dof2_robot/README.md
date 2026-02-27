# DOF2 Robot

Simple 2-DOF robot arm for simulation. Uses **primitive geometry only** (cylinder, box) in the URDF—no mesh files—so it works in Isaac Sim and RViz with no extra assets.

## Structure

- **joint1**: revolute (Z), base rotation
- **joint2**: revolute (Z), shoulder/elbow

Links: `base_link` (cylinder), `link1` (box), `link2` (box).

## Run in Isaac Sim

From docker_all (or repo root) with Isaac Sim venv active:

```bash
python components/ros2_comp/app/monty_demo/demos/dof2_isaac_demo.py
```

## URDF

- `urdf/dof2_robot.urdf` — single file, no `package://` or mesh references.

## Use in RViz

Point your robot_description or `robot_state_publisher` to `urdf/dof2_robot.urdf` (or the package-relative path). No mesh packages required.
