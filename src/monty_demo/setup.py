from setuptools import find_packages, setup

package_name = "monty_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={
        package_name: [
            "x3plus_isaac/urdf/*.urdf",
            "x3plus_isaac/meshes/**/*.STL",
            "x3plus_isaac/meshes/**/*.stl",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/x3plus_controllers.yaml"]),
        ("share/" + package_name + "/config", ["opus_plan_and_imp/config/opus_x3plus_controllers.yaml"]),
        ("share/" + package_name + "/launch", ["launch/x3plus_bringup.launch.py"]),
        ("share/" + package_name + "/launch", ["opus_plan_and_imp/launch/opus_x3plus_bringup.launch.py"]),
        (
            "share/" + package_name + "/urdf",
            ["x3plus_robot/urdf/x3plus.urdf.xacro", "x3plus_robot/urdf/ros2_control_topic.xacro"],
        ),
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
        ],
    },
)
