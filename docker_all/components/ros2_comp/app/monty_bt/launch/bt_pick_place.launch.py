"""Launch the BT pick-and-place executor node.

This is meant to be run *after* the main stack is up (real_up.sh).
The planner, MoveIt, and bridge must already be running.

Usage (from docker_all/):
  docker compose exec ros2_comp bash -l -c \
    "ros2 launch monty_bt bt_pick_place.launch.py"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "execute", default_value="true",
            description="true = move arm, false = plan only"),
        DeclareLaunchArgument(
            "dry_run", default_value="false",
            description="Safety lockout: forces execute=false"),
        DeclareLaunchArgument(
            "groot_port", default_value="0",
            description="Groot2 ZMQ monitor port (0 = disabled, 1667 = default Groot2 port)"),

        Node(
            package="monty_bt",
            executable="bt_pick_place_node",
            name="bt_pick_place",
            output="screen",
            parameters=[{
                "execute": LaunchConfiguration("execute"),
                "dry_run": LaunchConfiguration("dry_run"),
                "groot_port": LaunchConfiguration("groot_port"),
            }],
        ),
    ])
