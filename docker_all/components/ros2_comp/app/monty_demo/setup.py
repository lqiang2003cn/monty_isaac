import os
from setuptools import find_packages, setup

package_name = "monty_demo"


def get_x3plus_robot_mesh_data_files():
    """Install x3plus_robot/meshes under share/monty_demo/meshes preserving layout."""
    base = "x3plus_robot/meshes"
    for root, _dirs, files in os.walk(base):
        stls = [os.path.join(root, f) for f in files if f.lower().endswith(".stl")]
        if stls:
            # share/monty_demo/meshes/X3plus/visual, etc.
            rel = os.path.relpath(root, base)
            yield ("share/" + package_name + "/meshes/" + rel, stls)


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    # x3plus_isaac: single source at docker_all/shared/x3plus_isaac (mounted at /shared in Docker)
    package_data={package_name: []},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "opus_plan_and_imp/config/opus_x3plus_controllers.yaml",
                "x3plus_robot/config/x3plus.srdf",
                "x3plus_robot/config/kinematics.yaml",
                "x3plus_robot/config/joint_limits.yaml",
                "x3plus_robot/config/moveit_controllers.yaml",
                "x3plus_robot/config/ompl_planning.yaml",
            ],
        ),
        (
            "share/" + package_name + "/launch",
            [
                "opus_plan_and_imp/launch/opus_x3plus_bringup.launch.py",
                "opus_plan_and_imp/launch/x3plus_moveit.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/urdf",
            ["x3plus_robot/urdf/x3plus.urdf.xacro", "x3plus_robot/urdf/ros2_control_topic.xacro"],
        ),
        *list(get_x3plus_robot_mesh_data_files()),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="monty", 
    maintainer_email="user@example.com",
    description="Demo ROS2 package for monty_isaac",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "talker = monty_demo.talker_node:main",
            "listener = monty_demo.listener_node:main",
            "robot_description_publisher = monty_demo.robot_description_publisher_node:main",
            "opus_x3plus_real_bridge = monty_demo.opus_plan_and_imp.opus_x3plus_real_bridge:main",
            "opus_x3plus_zmq_bridge = monty_demo.opus_plan_and_imp.opus_x3plus_zmq_bridge:main",
            "x3plus_5dof_planner = monty_demo.x3plus_5dof_planner:main",
            "x3plus_reachability_test = monty_demo.x3plus_reachability_test:main",
            "straight_line_test = monty_demo.straight_line_test:main",
            "x3plus_joint_check = monty_demo.x3plus_joint_check:main",
            "x3plus_pick_place = monty_demo.x3plus_pick_place:main",
            "pick_place_test = monty_demo.pick_place_test:main",
        ],
    },
)
