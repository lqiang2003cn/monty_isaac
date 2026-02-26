# x3plus_isaac_2dof

Simplified X3plus for **Isaac Sim**: car base + first two arm joints only (2-DOF arm).

## Contents

- **urdf/x3plus_isaac_2dof.urdf** — Car base (base_link, camera, laser) + arm_link1, arm_link2
- **meshes/** — Same mesh layout as x3plus_isaac. See **meshes/README.md**.

## Run in Isaac Sim

From repo root, with meshes in place under `x3plus_isaac_2dof/meshes/`:

```bash
python src/monty_demo/demos/x3plus_isaac_2dof_demo.py
```
