# Meshes for x3plus_isaac

The URDF `../urdf/x3plus_isaac.urdf` references meshes with paths relative to `urdf/`, e.g. `../meshes/X3plus/visual/base_link.STL`.

**Setup:** Copy or symlink the mesh files so that this directory contains:

- `X3plus/visual/` and `X3plus/collision/` — base_link, arm_link1–5, rlink1–3, llink1–3 (gripper)
- `sensor/visual/` and `sensor/collision/` — camera_link, laser_link, mono_link

From the repo root (or from this folder):

```bash
# Copy from x3plus_robot in this repo
cp -r ../x3plus_robot/meshes/* ./

# Or symlink (if x3plus_robot has meshes)
ln -s ../x3plus_robot/meshes/X3plus ./
ln -s ../x3plus_robot/meshes/sensor ./
```

A full copy from `x3plus_robot/meshes` includes all arm and gripper meshes.
