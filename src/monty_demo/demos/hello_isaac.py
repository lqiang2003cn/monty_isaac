#!/usr/bin/env python3
"""
Hello Isaac Sim - Demo for Isaac Sim 5.1.0
Run with: python src/monty_demo/demos/hello_isaac.py

Isaac Sim 5.1.0 uses the new isaacsim API.
"""

print("=" * 50)
print("Starting Isaac Sim 5.1.0...")
print("=" * 50)

from isaacsim import SimulationApp

# Configuration
config = {
    "headless": False,  # Set to True for no GUI
    "width": 1280,
    "height": 720,
}

# Create simulation app
simulation_app = SimulationApp(config)  # pyright: ignore[reportOptionalCall]

print("\n" + "=" * 50)
print("Isaac Sim started successfully!")
print("=" * 50 + "\n")

# Import after SimulationApp is created - using Isaac Sim 5.1 API
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# Create world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add a falling cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="fancy_cube",
        position=np.array([0.0, 0.0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([0.2, 0.6, 0.9]),
    )
)

print("Created world with ground plane and cube")
print("Running simulation for 500 steps...")
print("Press Ctrl+C to exit\n")

# Reset and run
world.reset()

try:
    for step in range(500):
        world.step(render=True)
        if step % 100 == 0:
            print(f"Step {step}/500")
except KeyboardInterrupt:
    print("\nStopped by user")

print("\n" + "=" * 50)
print("Simulation complete!")
print("=" * 50)

# Cleanup
simulation_app.close()
