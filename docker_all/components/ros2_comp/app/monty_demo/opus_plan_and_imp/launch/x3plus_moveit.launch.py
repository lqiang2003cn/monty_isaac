#!/usr/bin/env python3
# MoveIt 2 launch for the X3plus 5-DOF arm.
#
# This launch file starts the move_group node with TRAC-IK (Distance mode) for
# the 5-DOF kinematic chain.  It is designed to be included from the main
# bringup launch (use_moveit:=true) or launched standalone after the bringup
# is already running.
#
# Standalone usage:
#   ros2 launch monty_demo x3plus_moveit.launch.py

import os
import re
import subprocess
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _strip_ros2_control(urdf_str: str) -> str:
    return re.sub(
        r"<ros2_control\b[^>]*>.*?</ros2_control>", "", urdf_str, flags=re.DOTALL
    )


def _flatten(d, parent_key="", sep="."):
    """Recursively flatten a nested dict into dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _write_params_yaml(params_dict):
    """Write a proper YAML parameter file without !!python/tuple tags."""
    flat = _flatten(params_dict)
    param_structure = {"/**": {"ros__parameters": flat}}
    fd, path = tempfile.mkstemp(prefix="moveit_params_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(param_structure, f, default_flow_style=False)
    return path


def generate_launch_description():
    pkg_dir = get_package_share_directory("monty_demo")
    urdf_dir = os.path.join(pkg_dir, "urdf")
    config_dir = os.path.join(pkg_dir, "config")

    # ── Robot description (same xacro as bringup) ──
    urdf_xacro = os.path.join(urdf_dir, "x3plus.urdf.xacro")
    robot_description_full = subprocess.check_output(
        ["xacro", urdf_xacro, "robot_name:=x3plus"], text=True, cwd=urdf_dir
    )
    robot_description_urdf_only = _strip_ros2_control(robot_description_full)

    # ── SRDF ──
    srdf_path = os.path.join(config_dir, "x3plus.srdf")
    with open(srdf_path) as f:
        srdf_content = f.read()

    # ── Config YAMLs ──
    kinematics_yaml = _load_yaml(os.path.join(config_dir, "kinematics.yaml"))
    joint_limits_yaml = _load_yaml(os.path.join(config_dir, "joint_limits.yaml"))
    moveit_controllers_yaml = _load_yaml(
        os.path.join(config_dir, "moveit_controllers.yaml")
    )
    ompl_yaml = _load_yaml(os.path.join(config_dir, "ompl_planning.yaml"))

    # ── Assemble all move_group parameters ──
    all_params = {
        "robot_description": robot_description_urdf_only,
        "robot_description_semantic": srdf_content,
        "robot_description_kinematics": kinematics_yaml,
        "robot_description_planning": joint_limits_yaml,
        "use_sim_time": False,
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_yaml,
        "trajectory_execution": {
            "allowed_execution_duration_scaling": 2.0,
            "allowed_goal_duration_margin": 0.5,
            "allowed_start_tolerance": 0.1,
        },
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    all_params.update(moveit_controllers_yaml)

    # Write to a clean YAML file (avoids !!python/tuple from launch_ros)
    params_file_path = _write_params_yaml(all_params)

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[params_file_path],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz", default_value="false", description="Launch RViz with MoveIt"
        ),
        move_group_node,
    ])
