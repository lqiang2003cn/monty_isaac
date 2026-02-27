# Meshes for x3plus_isaac_2dof

The URDF `../urdf/x3plus_isaac_2dof.urdf` references meshes with paths relative to `urdf/`, e.g. `../meshes/X3plus/visual/base_link.STL`.

**Setup:** Copy or symlink mesh files so that this directory contains:

- `X3plus/visual/` and `X3plus/collision/` — base_link, arm_link1, arm_link2
- `sensor/visual/` and `sensor/collision/` — camera_link, laser_link

From the repo root (or from this folder):

```bash
# From x3plus_isaac_2dof/meshes/ — symlink from x3plus_isaac (if meshes already set up there)
ln -sf ../../monty_demo/x3plus_isaac/meshes/X3plus ./
ln -sf ../../monty_demo/x3plus_isaac/meshes/sensor ./

# Or copy from x3plus_robot
cp -r ../../x3plus_robot/meshes/X3plus ./
cp -r ../../x3plus_robot/meshes/sensor ./
```
