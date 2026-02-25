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

**Important**: Use Isaac Sim's Python launcher to run scripts:

```bash
# From workspace root
cd /workspaces/isaac_ros_ws

# Hello World demo
/isaac-sim/python.sh src/project/demos/hello_isaac.py

# Extended demo
/isaac-sim/python.sh src/project/demos/isaac_sim_demo.py

# ROS2 integration demo
/isaac-sim/python.sh src/project/demos/isaac_sim_ros2_demo.py

# Or use the alias (after rebuilding container)
python-isaac src/project/demos/hello_isaac.py
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
