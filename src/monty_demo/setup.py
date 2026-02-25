from setuptools import find_packages, setup

package_name = "monty_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
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
        ],
    },
)
