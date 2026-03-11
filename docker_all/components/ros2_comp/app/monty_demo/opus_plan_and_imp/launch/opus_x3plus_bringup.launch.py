#!/usr/bin/env python3
# Unified bringup for x3plus ros2_control (Isaac Sim, real robot, or ZMQ).
# mode:=isaac -> user runs Isaac bridge separately (needs Isaac venv).
# mode:=real  -> launches opus_x3plus_real_bridge node (local USB serial).
# mode:=zmq   -> launches opus_x3plus_zmq_bridge node (remote robot via ZMQ).
#
# use_camera:=true -> launches Intel RealSense RGBD camera via realsense2_camera.
# use_rviz:=true   -> launches RViz2 with camera + robot display.
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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

_CAMERA_AVAILABLE = False
_CAMERA_PKG_DIR = ""
try:
    _CAMERA_PKG_DIR = get_package_share_directory("realsense2_camera")
    _CAMERA_AVAILABLE = True
except Exception:
    pass


def _strip_ros2_control(urdf_str: str) -> str:
    """Remove <ros2_control>...</ros2_control> so robot_state_publisher can parse."""
    return re.sub(r"<ros2_control\b[^>]*>.*?</ros2_control>", "", urdf_str, flags=re.DOTALL)


def generate_launch_description():
    pkg_dir = get_package_share_directory("monty_demo")
    urdf_dir = os.path.join(pkg_dir, "urdf")
    urdf_xacro = os.path.join(pkg_dir, "urdf", "x3plus.urdf.xacro")
    controllers_yaml = os.path.join(pkg_dir, "config", "opus_x3plus_controllers.yaml")
    rviz_config = os.path.join(pkg_dir, "config", "camera_view.rviz")

    robot_description_full = subprocess.check_output(
        ["xacro", urdf_xacro, "robot_name:=x3plus"],
        text=True,
        cwd=urdf_dir,
    )
    robot_description_urdf_only = _strip_ros2_control(robot_description_full)

    mode_is_real = PythonExpression(["'", LaunchConfiguration("mode"), "' == 'real'"])
    mode_is_zmq = PythonExpression(["'", LaunchConfiguration("mode"), "' == 'zmq'"])
    use_moveit = LaunchConfiguration("use_moveit")
    use_bt = LaunchConfiguration("use_bt")
    use_camera = LaunchConfiguration("use_camera")
    use_rviz = LaunchConfiguration("use_rviz")
    use_monty = LaunchConfiguration("use_monty")
    debug_logs = LaunchConfiguration("debug_logs")
    log_level = PythonExpression([
        "'debug' if '", debug_logs, "' == 'true' else 'info'",
    ])

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "x3plus_moveit.launch.py")
        ),
        condition=IfCondition(use_moveit),
    )

    actions = [
        DeclareLaunchArgument("mode", default_value="isaac", description="isaac, real, or zmq"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0", description="Serial port for real robot (mode:=real)"),
        DeclareLaunchArgument("zmq_host", default_value="192.168.31.142", description="ZMQ service host (mode:=zmq)"),
        DeclareLaunchArgument("zmq_port", default_value="5555", description="ZMQ service port (mode:=zmq)"),
        DeclareLaunchArgument("use_moveit", default_value="false", description="Launch MoveIt move_group"),
        DeclareLaunchArgument("debug_logs", default_value="false", description="Enable debug-level ROS console logging on planner and bridge nodes (noisy; prefer file logs)"),
        DeclareLaunchArgument("use_bt", default_value="false", description="Launch the BT pick-place executor node"),
        DeclareLaunchArgument("use_camera", default_value="false", description="Launch Intel RealSense RGBD camera"),
        DeclareLaunchArgument("use_rviz", default_value="false", description="Launch RViz2 with camera + robot view"),
        DeclareLaunchArgument("use_monty", default_value="false", description="Launch scan control bridge for Monty turntable scanning"),
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
            ros_arguments=["--log-level", log_level],
        ),
        Node(
            package="monty_demo",
            executable="opus_x3plus_zmq_bridge",
            name="opus_x3plus_zmq_bridge",
            output="screen",
            condition=IfCondition(mode_is_zmq),
            parameters=[
                {"zmq_host": LaunchConfiguration("zmq_host")},
                {"zmq_port": LaunchConfiguration("zmq_port")},
            ],
            ros_arguments=["--log-level", log_level],
        ),
        moveit_launch,
        Node(
            package="monty_demo",
            executable="x3plus_5dof_planner",
            name="x3plus_5dof_planner",
            output="screen",
            condition=IfCondition(use_moveit),
            ros_arguments=["--log-level", log_level],
        ),
        Node(
            package="monty_bt",
            executable="bt_pick_place_node",
            name="bt_pick_place",
            output="screen",
            condition=IfCondition(use_bt),
        ),
    ]

    if _CAMERA_AVAILABLE:
        _rs_launch = os.path.join(_CAMERA_PKG_DIR, "launch", "rs_launch.py")
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(_rs_launch),
                launch_arguments={
                    "camera_name": "cam",
                    "camera_namespace": "",
                    "pointcloud.enable": "true",
                    "align_depth.enable": "true",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "initial_reset": "true",
                }.items(),
                condition=IfCondition(use_camera),
            )
        )
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="cam_tf",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "camera_link",
                    "--child-frame-id", "cam_link",
                ],
                condition=IfCondition(use_camera),
            )
        )
    else:
        actions.append(
            LogInfo(
                msg="realsense2_camera package not found — camera launch skipped",
                condition=IfCondition(use_camera),
            )
        )

    actions.append(
        Node(
            package="monty_demo",
            executable="scan_control_bridge",
            name="scan_control_bridge",
            output="screen",
            condition=IfCondition(use_monty),
        ),
    )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
        ),
    )

    return LaunchDescription(actions)
