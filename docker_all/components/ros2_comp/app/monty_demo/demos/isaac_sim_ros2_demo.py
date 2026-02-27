#!/usr/bin/env python3
"""
Isaac Sim + ROS2 Integration Demo

Local installation:
    source ~/isaacsim_venv/bin/activate
    export ACCEPT_EULA=Y
    python src/monty_demo/demos/isaac_sim_ros2_demo.py

This demo shows how to start Isaac Sim with ROS2 bridge enabled,
create a simple scene with objects, and publish to ROS2 topics.

NOTE: This demo does NOT require NVIDIA Nucleus server connection.
It uses only local primitives and the ROS2 clock publisher.

Isaac Sim 5.1.0 API
"""

# IMPORTANT: isaacsim must be imported before any other omniverse imports
from isaacsim import SimulationApp
import os

# Configuration for the simulation with ROS2 support
CONFIG = {
    "headless": False,  # Set to True for headless mode (no GUI)
    "width": 1280,
    "height": 720,
}

ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
rmw_impl = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
pythonpath = os.environ.get("PYTHONPATH", "")
print("Starting Isaac Sim with ROS2 support...")
print(f"ROS_DOMAIN_ID={ros_domain_id}")
print(f"ROS_DISTRO={ros_distro}")
print(f"RMW_IMPLEMENTATION={rmw_impl}")
print(f"PYTHONPATH={pythonpath}")
simulation_app = SimulationApp(CONFIG)  # pyright: ignore[reportOptionalCall]
print("Isaac Sim started!")

# Import omniverse modules after SimulationApp is created
# Using Isaac Sim 5.1 API
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.extensions import enable_extension
from omni.kit.app import get_app
import numpy as np

# Enable ROS2 Bridge extension
print("Enabling ROS2 Bridge extension...")
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.ros2.bridge")
ext_manager = get_app().get_extension_manager()
for _ in range(20):
    simulation_app.update()
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        break
print("✓ ROS2 Bridge extension enabled")

# Verify rclpy loaded
try:
    import rclpy
    print("✓ rclpy imported successfully")
except ImportError as e:
    print(f"✗ Could not import rclpy: {e}")
    print("ROS2 topics will NOT be published!")


def create_simple_ros2_scene():
    """
    Create a simple scene with physics objects.
    The ROS2 bridge automatically publishes /clock when enabled.
    """
    
    world = World(stage_units_in_meters=1.0)
    
    # Add ground plane
    world.scene.add_default_ground_plane()
    print("Added ground plane")
    
    # Add some dynamic cubes that will fall and interact
    colors = [
        [1.0, 0.2, 0.2],  # Red
        [0.2, 1.0, 0.2],  # Green  
        [0.2, 0.2, 1.0],  # Blue
        [1.0, 1.0, 0.2],  # Yellow
    ]
    
    for i, color in enumerate(colors):
        cube = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Cube_{i}",
                name=f"cube_{i}",
                position=np.array([i * 0.3 - 0.45, 0.0, 1.0 + i * 0.5]),
                scale=np.array([0.2, 0.2, 0.2]),
                color=np.array(color),
            )
        )
    print(f"Added {len(colors)} dynamic cubes")
    
    # Add a static obstacle
    obstacle = world.scene.add(
        VisualCuboid(
            prim_path="/World/Obstacle",
            name="obstacle",
            position=np.array([0.0, 0.0, 0.1]),
            scale=np.array([1.0, 1.0, 0.2]),
            color=np.array([0.5, 0.5, 0.5]),
        )
    )
    print("Added static obstacle")
    
    return world


def setup_ros2_clock_publisher():
    """
    Set up ROS2 clock publisher using OmniGraph.
    This publishes simulation time to /clock topic.
    """
    import omni.graph.core as og
    
    try:
        domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
        # Create an OmniGraph for ROS2 clock using correct node types
        keys = og.Controller.Keys
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": "/World/ROS2_Clock", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                ],
            },
        )
        print(f"✓ ROS2 Clock publisher created on /clock topic (domain {domain_id})")
        print("  (ROS2Context will use ROS_DOMAIN_ID from environment)")
        return True
    except Exception as e:
        print(f"✗ Could not create clock publisher: {e}")
        return False


def run_ros2_simulation(world, num_steps=None):
    """Run the simulation with ROS2 integration."""
    
    if num_steps:
        print(f"\nStarting ROS2 simulation for {num_steps} steps...")
    else:
        print("\nStarting ROS2 simulation (infinite loop)...")
    print("In another terminal, try:")
    print("  ros2 topic list")
    print("  ros2 topic echo /clock")
    print("\nPress Ctrl+C to stop\n")
    
    world.reset()
    try:
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
    except Exception as e:
        print(f"Could not start timeline: {e}")
    
    step = 0
    try:
        while True:
            world.step(render=True)
            
            if step % 200 == 0:
                if num_steps:
                    print(f"Simulation step: {step}/{num_steps}")
                else:
                    print(f"Simulation step: {step}")
            
            step += 1
            
            if num_steps and step >= num_steps:
                break
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    print("ROS2 simulation complete!")


def main():
    """Main function to run the ROS2 demo."""
    
    print("\n" + "="*50)
    print("Isaac Sim + ROS2 Demo")
    print("="*50 + "\n")
    
    # Create scene
    world = create_simple_ros2_scene()
    
    # Set up ROS2 clock publisher
    setup_ros2_clock_publisher()
    
    # Run simulation (infinite loop until Ctrl+C)
    run_ros2_simulation(world)
    
    # Cleanup
    print("\nClosing Isaac Sim...")
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
