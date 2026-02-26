#!/usr/bin/env python3
# Unified bringup for x3plus ros2_control (Isaac Sim or real robot).
# mode:=isaac -> user runs Isaac bridge separately (needs Isaac venv).
# mode:=real  -> launches opus_x3plus_real_bridge node.
#
# use_sim_time is always false: the topic-based hardware interface uses
# wall-clock timestamps and does not require a /clock publisher. Isaac Sim's
# rclpy bridge has a Python version mismatch (3.11 vs 3.12) that prevents
# /clock publication, so sim time would cause controller activation to hang.
#
# Prerequisite: xacro on PATH (ros-jazzy-xacro).

import os
import re
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _strip_ros2_control(urdf_str: str) -> str:
    """Remove <ros2_control>...</ros2_control> so robot_state_publisher can parse."""
    return re.sub(r"<ros2_control\b[^>]*>.*?</ros2_control>", "", urdf_str, flags=re.DOTALL)


def generate_launch_description():
    pkg_dir = get_package_share_directory("monty_demo")
    urdf_dir = os.path.join(pkg_dir, "urdf")
    urdf_xacro = os.path.join(pkg_dir, "urdf", "x3plus.urdf.xacro")
    controllers_yaml = os.path.join(pkg_dir, "config", "opus_x3plus_controllers.yaml")

    robot_description_full = subprocess.check_output(
        ["xacro", urdf_xacro],
        text=True,
        cwd=urdf_dir,
    )
    robot_description_urdf_only = _strip_ros2_control(robot_description_full)

    mode_is_real = PythonExpression(["'", LaunchConfiguration("mode"), "' == 'real'"])

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="isaac", description="isaac or real"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0", description="Serial port for real robot"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": robot_description_urdf_only},
                {"use_sim_time": False},
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
                {"use_sim_time": False},
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
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            name="joint_trajectory_controller_spawner",
            arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="monty_demo",
            executable="opus_x3plus_real_bridge",
            name="opus_x3plus_real_bridge",
            output="screen",
            condition=IfCondition(mode_is_real),
            parameters=[
                {"serial_port": LaunchConfiguration("serial_port")},
            ],
        ),
    ])
