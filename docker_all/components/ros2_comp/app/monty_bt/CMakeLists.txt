cmake_minimum_required(VERSION 3.14)
project(monty_bt)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(behaviortree_ros2 REQUIRED)

add_executable(bt_pick_place_node
  src/bt_pick_place_node.cpp
)
target_include_directories(bt_pick_place_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(bt_pick_place_node
  rclcpp
  std_srvs
  rcl_interfaces
  ament_index_cpp
  behaviortree_cpp
  behaviortree_ros2
)

install(TARGETS bt_pick_place_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY trees/
  DESTINATION share/${PROJECT_NAME}/trees
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
