#!/usr/bin/env python3
"""
Isaac Sim Demo - Basic simulation startup and scene creation
Run with: /isaac-sim/python.sh src/project/demos/isaac_sim_demo.py

This demo shows how to start Isaac Sim, create a simple scene with a robot,
and run the simulation loop.

Isaac Sim 5.1.0 API
"""

# IMPORTANT: isaacsim must be imported before any other omniverse imports
from isaacsim import SimulationApp

# Configuration for the simulation
CONFIG = {
    "headless": False,  # Set to True for headless mode (no GUI)
    "width": 1280,
    "height": 720,
    "anti_aliasing": 0,  # 0 = off, 1 = FXAA, 2 = DLSS
    "renderer": "RayTracedLighting",  # Options: "RayTracedLighting", "PathTracing"
}

# Create the simulation application
print("Starting Isaac Sim...")
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]
print("Isaac Sim started successfully!")

# Now we can import omniverse modules after SimulationApp is created
# Using Isaac Sim 5.1 API
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np


def create_simple_scene():
    """Create a simple scene with ground plane and some objects."""
    
    # Create the world
    world = World(stage_units_in_meters=1.0)
    
    # Add a ground plane
    world.scene.add_default_ground_plane()
    print("Added ground plane")
    
    # Add a dynamic cube that will fall due to gravity
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="fancy_cube",
            position=np.array([0.0, 0.0, 1.0]),  # Start 1 meter above ground
            scale=np.array([0.5, 0.5, 0.5]),  # 0.5m cube
            color=np.array([0.2, 0.6, 0.9]),  # Light blue color
        )
    )
    print("Added dynamic cube")
    
    return world


def create_robot_scene():
    """Create a scene with a Franka robot using Isaac Sim's Franka class."""
    
    # Create the world
    world = World(stage_units_in_meters=1.0)
    
    # Add ground plane
    world.scene.add_default_ground_plane()
    
    # Try to use the Franka robot class which handles asset loading properly
    try:
        # Import the Franka robot class from Isaac Sim
        from isaacsim.robot.manipulators.examples.franka import Franka
        
        # Add Franka robot using the built-in class
        franka = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka_robot",
                position=np.array([0.0, 0.0, 0.0]),
            )
        )
        print("Added Franka robot using built-in Franka class")
        
    except ImportError as e:
        print(f"Franka class not available: {e}")
        print("Trying alternative method...")
        
        # Fallback: Try using UR10 or another robot
        try:
            from isaacsim.robot.manipulators.examples.universal_robots import UR10
            
            robot = world.scene.add(
                UR10(
                    prim_path="/World/UR10",
                    name="ur10_robot",
                    position=np.array([0.0, 0.0, 0.0]),
                )
            )
            print("Added UR10 robot as fallback")
        except Exception as e2:
            print(f"UR10 also failed: {e2}")
            print("Creating colored cubes as visual fallback...")
            
            # Create some interesting objects instead
            for i in range(5):
                cube = world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/Cube_{i}",
                        name=f"cube_{i}",
                        position=np.array([i * 0.3 - 0.6, 0.0, 0.5 + i * 0.2]),
                        scale=np.array([0.15, 0.15, 0.15]),
                        color=np.array([0.2 + i * 0.15, 0.3, 0.8 - i * 0.1]),
                    )
                )
            print("Created 5 colored falling cubes")
    
    except Exception as e:
        print(f"Could not load robot: {e}")
        print("Creating fallback cubes...")
        for i in range(3):
            cube = world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{i}",
                    name=f"cube_{i}",
                    position=np.array([i * 0.4 - 0.4, 0.0, 1.0]),
                    scale=np.array([0.2, 0.2, 0.2]),
                    color=np.array([1.0 - i * 0.3, i * 0.3, 0.5]),
                )
            )
    
    return world


def run_simulation(world, num_steps=None):
    """Run the simulation. If num_steps is None, run indefinitely."""
    
    if num_steps:
        print(f"\nStarting simulation for {num_steps} steps...")
    else:
        print("\nStarting simulation (infinite loop)...")
    print("Press Ctrl+C to stop")
    print("TIP: Use mouse to navigate - Right-click+drag to rotate, Middle-click+drag to pan, Scroll to zoom\n")
    
    # Reset the world before starting
    world.reset()
    
    # Frame all content in viewport after first reset
    try:
        import omni.kit.viewport.utility as viewport_utils
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            # Frame all prims in the scene
            import omni.kit.commands
            omni.kit.commands.execute("FrameAllPrims")
            print("Viewport framed to show all objects")
    except Exception as e:
        print(f"Could not frame viewport: {e}")
    
    step = 0
    try:
        while True:
            # Step the simulation
            world.step(render=True)
            
            # Print progress every 500 steps
            if step % 500 == 0:
                print(f"Simulation step: {step}")
            
            step += 1
            
            # Break if we have a step limit
            if num_steps and step >= num_steps:
                break
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print("Simulation complete!")


def main():
    """Main function to run the demo."""
    
    print("\n" + "="*50)
    print("Isaac Sim Demo")
    print("="*50 + "\n")
    
    # Option 1: Simple scene with falling cube (commented out)
    # print("Creating simple scene with dynamic cube...")
    # world = create_simple_scene()
    
    # Option 2: Scene with robot
    print("Creating scene with robot...")
    world = create_robot_scene()
    
    # Run the simulation (infinite loop until Ctrl+C)
    run_simulation(world)
    
    # Cleanup
    print("\nClosing Isaac Sim...")
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
