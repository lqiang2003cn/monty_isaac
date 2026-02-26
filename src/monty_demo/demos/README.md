# Isaac Sim 5.1.0 Demo Scripts

This folder contains Python demos for NVIDIA Isaac Sim 5.1.0.

## Environment

- **Isaac Sim**: 5.1.0 (local install)
- **Python**: 3.11 (bundled with Isaac Sim or venv)
- **ROS2**: Jazzy (or as installed)
- **Documentation**: https://docs.isaacsim.omniverse.nvidia.com/latest/

## Prerequisites

1. Install Isaac Sim locally and create a virtual environment (see [Isaac Sim Python tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/python_tutorials.html)).
2. Activate your Isaac Sim venv before running demos.

## Running Demos

From the **repository root** with your Isaac Sim virtualenv activated:

```bash
python src/monty_demo/demos/hello_isaac.py
python src/monty_demo/demos/isaac_sim_demo.py
python src/monty_demo/demos/isaac_sim_ros2_demo.py
python src/monty_demo/demos/x3plus_isaac_demo.py
python src/monty_demo/demos/dof2_isaac_demo.py
python src/monty_demo/demos/x3plus_isaac_2dof_demo.py
```

Or run the ROS2-control arm demo as a module:

```bash
python -m monty_demo.x3plus_isaac_arm_demo
```

## Demo Scripts

### 1. hello_isaac.py - Basic Demo

Shows:
- Starting Isaac Sim with GUI
- Creating a world with physics
- Adding a falling cube
- Running the simulation loop

### 2. isaac_sim_demo.py - Extended Demo

Shows:
- Different scene configurations
- Loading robots from Nucleus
- Simulation control

### 3. isaac_sim_ros2_demo.py - ROS2 Integration

Shows:
- Enabling ROS2 Bridge extension
- Publishing simulation clock to `/clock`
- Integrating with ROS2 ecosystem

### 4. x3plus_isaac_demo.py - X3plus in Isaac Sim

Loads the X3plus robot from `src/monty_demo/x3plus_robot/urdf/x3plus.urdf`, resolves ROS `package://` mesh paths, and runs the simulation.

**Meshes required.** The URDF references meshes via `package://lqtech_ros2_x3plus/meshes/...`. Do one of the following:

- **Copy meshes into the repo:** Copy (or symlink) the `meshes` folder from the **lqtech_ros2_x3plus** package into `src/monty_demo/x3plus_robot/meshes/`, keeping the same structure (`X3plus/visual`, `X3plus/collision`, `sensor/visual`, `sensor/collision`). See `src/monty_demo/x3plus_robot/meshes/README.md`.
- **Use an existing package path:** Set `X3PLUS_MESH_ROOT` to the root of the package that contains the `meshes` directory (e.g. the full path to `lqtech_ros2_x3plus`), then run the demo.

If the URDF importer reports missing commands, enable the extension **isaacsim.asset.importer.urdf** via Window > Extensions (it is usually loaded by default).

**RViz vs Isaac Sim display:** The same URDF can look different because (1) RViz uses `/joint_states` and robot_state_publisher to compute mimic joints from the source joint; (2) Isaac Sim runs physics and requires mimic follower joints to have finite limits (the demo rewrites continuous mimic joints to revolute with limits); (3) default joint positions may differ—the demo sets initial positions to 0 after reset to match RViz. Both use Z-up and meters; the demo sets `distance_scale=1` for the importer.

### 5. dof2_isaac_demo.py - 2-DOF robot arm

Loads the simple 2-DOF arm from `src/monty_demo/dof2_robot/urdf/dof2_robot.urdf`. The robot uses **primitive geometry only** (cylinder, box)—no mesh files—so it runs in Isaac Sim with no extra setup.

### 6. x3plus_isaac_arm_demo.py - X3plus in Isaac Sim (arm + gripper)

In `monty_demo/monty_demo/x3plus_isaac_arm_demo.py`. Loads `src/monty_demo/x3plus_isaac/urdf/x3plus_isaac.urdf`: single URDF for Isaac with relative mesh paths (`../meshes/...`), arm + base + sensors + gripper. Publishes/subscribes to `/x3plus/joint_states` and `/x3plus/joint_commands` for ros2_control (JointStateTopicSystem). Put meshes under `monty_demo/x3plus_isaac/meshes/` (copy or symlink from `x3plus_robot/meshes`); see `monty_demo/x3plus_isaac/meshes/README.md`.

## Running Isaac Sim Directly

Use your local Isaac Sim install, for example:

```bash
# If using standalone app
~/isaac-sim/isaac-sim.sh
~/isaac-sim/isaac-sim.sh --headless
~/isaac-sim/python.sh

# If using venv
source ~/isaacsim_venv/bin/activate
python ...
```

## API Reference

Isaac Sim 5.1.0 uses the `isaacsim` API:

| Module | Description |
|--------|-------------|
| `isaacsim.core.api` | Core simulation API (World, Scene) |
| `isaacsim.core.api.objects` | Physics objects (DynamicCuboid, etc.) |
| `isaacsim.core.api.prims` | USD primitives |
| `isaacsim.storage.native` | Asset management |
| `isaacsim.core.utils` | Utility functions |

## Troubleshooting

### No GUI Window

- Check DISPLAY: `echo $DISPLAY`
- Test X11: `xeyes`

### Slow First Launch

First launch may load shaders and caches. Subsequent launches are faster.

### Module Not Found

Use your Isaac Sim Python (venv or Isaac Sim’s bundled Python), not the system `python3`.

## Resources

- [Isaac Sim 5.1 Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/)
- [Python Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/python_tutorials.html)
- [ROS2 Bridge](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials.html)
