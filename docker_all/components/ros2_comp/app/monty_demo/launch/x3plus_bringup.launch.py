#!/usr/bin/env python3
# Launch controller_manager with x3plus robot description (JointStateTopicSystem).
# Start Isaac Sim demo (or real robot driver) first so /x3plus/joint_states and
# /x3plus/joint_commands are active.
#
# Prerequisite: sudo apt install ros-jazzy-xacro  (so the 'xacro' command is on PATH)

import os
import re
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _strip_ros2_control(urdf_str: str) -> str:
    """Remove <ros2_control>...</ros2_control> so robot_state_publisher can parse (no joint type error)."""
    return re.sub(r"<ros2_control\b[^>]*>.*?</ros2_control>", "", urdf_str, flags=re.DOTALL)


def generate_launch_description():
    pkg_dir = get_package_share_directory("monty_demo")
    urdf_dir = os.path.join(pkg_dir, "urdf")
    urdf_xacro = os.path.join(pkg_dir, "urdf", "x3plus.urdf.xacro")
    controllers_yaml = os.path.join(pkg_dir, "config", "x3plus_controllers.yaml")

    # Expand xacro once (full); robot_state_publisher gets URDF with ros2_control stripped
    robot_description_full = subprocess.check_output(
        ["xacro", urdf_xacro],
        text=True,
        cwd=urdf_dir,
    )
    robot_description_urdf_only = _strip_ros2_control(robot_description_full)

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use sim time (true for Isaac Sim)"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": robot_description_urdf_only},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
        Node(
            package="monty_demo",
            executable="robot_description_publisher",
            name="robot_description_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description_full}],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            name="controller_manager",
            output="screen",
            parameters=[
                controllers_yaml,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            remappings=[
                ("/robot_description", "/robot_description_full"),
            ],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            name="joint_state_broadcaster_spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            name="joint_trajectory_controller_spawner",
            arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
            output="screen",
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
