# Isaac Sim 5.1.0 Demo Scripts

This folder contains Python demos for NVIDIA Isaac Sim 5.1.0.

## Environment

- **Isaac Sim Version**: 5.1.0 (official NVIDIA container)
- **Base Image**: `nvcr.io/nvidia/isaac-sim:5.1.0`
- **Python**: 3.11 (bundled with Isaac Sim)
- **ROS2**: Humble
- **Documentation**: https://docs.isaacsim.omniverse.nvidia.com/latest/

## Prerequisites

### 1. NGC Authentication (required for first build)

On your **host machine**, log in to NVIDIA NGC:

```bash
# Get API key from https://ngc.nvidia.com/setup/api-key
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### 2. Create Cache Directories

```bash
mkdir -p ~/docker/isaac-sim/cache/kit
mkdir -p ~/docker/isaac-sim/cache/ov
mkdir -p ~/docker/isaac-sim/cache/pip
mkdir -p ~/docker/isaac-sim/cache/glcache
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/data
```

### 3. Enable X11 Display

```bash
xhost +local:docker
```

## Running Demos

**Important**: Run scripts with Isaac Sim's Python (either inside the container or your local Isaac Sim venv).

### Inside Isaac Sim container

From the workspace root inside the container (e.g. `/workspaces/isaac_ros_ws`):

```bash
/isaac-sim/python.sh src/project/demos/hello_isaac.py
/isaac-sim/python.sh src/project/demos/isaac_sim_demo.py
/isaac-sim/python.sh src/project/demos/isaac_sim_ros2_demo.py
/isaac-sim/python.sh src/project/demos/x3plus_isaac_demo.py
/isaac-sim/python.sh src/project/demos/dof2_isaac_demo.py
/isaac-sim/python.sh src/project/demos/x3plus_isaac_arm_demo.py
# Or: python-isaac src/project/demos/hello_isaac.py
```

### On host with Isaac Sim venv

From the **repository root** (e.g. `~/lqtech/dockers/monty_isaac`) with your Isaac Sim virtualenv activated:

```bash
python src/monty_demo/demos/hello_isaac.py
python src/monty_demo/demos/x3plus_isaac_demo.py
python src/monty_demo/demos/dof2_isaac_demo.py
python src/monty_demo/demos/x3plus_isaac_arm_demo.py
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

Loads `src/monty_demo/x3plus_isaac/urdf/x3plus_isaac.urdf`: single URDF for Isaac with relative mesh paths (`../meshes/...`), arm + base + sensors + gripper. Gripper mimic joints use revolute limits so they display correctly. Put meshes under `x3plus_isaac/meshes/` (copy or symlink from `x3plus_robot/meshes`); see `x3plus_isaac/meshes/README.md`.

## Running Isaac Sim Directly

```bash
# Full GUI application
/isaac-sim/isaac-sim.sh

# Headless mode
/isaac-sim/isaac-sim.sh --headless

# Python REPL with Isaac Sim
/isaac-sim/python.sh
```

## API Reference

Isaac Sim 5.1.0 uses the new `isaacsim` API:

| Module | Description |
|--------|-------------|
| `isaacsim.core.api` | Core simulation API (World, Scene) |
| `isaacsim.core.api.objects` | Physics objects (DynamicCuboid, etc.) |
| `isaacsim.core.api.prims` | USD primitives |
| `isaacsim.storage.native` | Asset management |
| `isaacsim.core.utils` | Utility functions |

## Troubleshooting

### Container Build Fails

1. Ensure you're logged into NGC: `docker login nvcr.io`
2. Check your NGC API key is valid

### No GUI Window

1. On host: `xhost +local:docker`
2. Check DISPLAY: `echo $DISPLAY`
3. Test X11: `xeyes`

### Slow First Launch

First launch downloads shaders and caches data. Subsequent launches are faster (~10-15 seconds).

### Module Not Found

Always use `/isaac-sim/python.sh` instead of `python3` to ensure correct Python environment.

## Resources

- [Isaac Sim 5.1 Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/)
- [Container Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Python Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/python_tutorials.html)
- [ROS2 Bridge](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials.html)
