#pragma once

#include <cmath>
#include <string>
#include <behaviortree_cpp/bt_factory.h>

namespace monty_bt
{

// J5 limits from x3plus_5dof_planner.py JOINT_LIMITS["arm_joint5"]
constexpr double kJ5Lo = -M_PI / 2.0;           // -1.5708
constexpr double kJ5Hi = M_PI;                   // 3.14159
constexpr double kMaxTiltRad = 5.0 * M_PI / 180; // 5 degrees
constexpr double kEps = 1e-6;

inline void quaternion_to_rpy(double qx, double qy, double qz, double qw,
                              double& roll, double& pitch, double& yaw)
{
  double sinr_cosp = 2.0 * (qw * qx + qy * qz);
  double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
  roll = std::atan2(sinr_cosp, cosr_cosp);

  double sinp = std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0);
  pitch = std::asin(sinp);

  double siny_cosp = 2.0 * (qw * qz + qx * qy);
  double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  yaw = std::atan2(siny_cosp, cosy_cosp);
}

inline bool map_yaw_to_j5(double yaw, double& j5_out)
{
  for (double candidate : {yaw, yaw - M_PI, yaw + M_PI})
  {
    if (candidate >= kJ5Lo - kEps && candidate <= kJ5Hi + kEps)
    {
      j5_out = candidate;
      return true;
    }
  }
  return false;
}

/**
 * Pure-math BT node: computes z_grasp, z_approach, and j5 (wrist roll)
 * from a block's 6-DOF surface pose, exactly matching the Python
 * parse_block_pose / z-height logic in x3plus_pick_place.py.
 */
class ComputeGraspParams : public BT::SyncActionNode
{
public:
  ComputeGraspParams(const std::string& name, const BT::NodeConfig& config)
      : SyncActionNode(name, config)
  {}

  static BT::PortsList providedPorts()
  {
    return {
        BT::InputPort<double>("bx"),
        BT::InputPort<double>("by"),
        BT::InputPort<double>("bz"),
        BT::InputPort<double>("qx", 0.0, "Quaternion x"),
        BT::InputPort<double>("qy", 0.0, "Quaternion y"),
        BT::InputPort<double>("qz", 0.0, "Quaternion z"),
        BT::InputPort<double>("qw", 1.0, "Quaternion w"),
        BT::InputPort<double>("approach_height", 0.04, "Clearance above grasp"),
        BT::InputPort<double>("grasp_depth", 0.015, "Depth below block surface"),
        BT::OutputPort<double>("z_grasp"),
        BT::OutputPort<double>("z_approach"),
        BT::OutputPort<double>("j5"),
    };
  }

  BT::NodeStatus tick() override
  {
    double bz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
    double approach_height = 0.04, grasp_depth = 0.015;

    if (!getInput("bz", bz))
      return BT::NodeStatus::FAILURE;
    getInput("qx", qx);
    getInput("qy", qy);
    getInput("qz", qz);
    getInput("qw", qw);
    getInput("approach_height", approach_height);
    getInput("grasp_depth", grasp_depth);

    double roll, pitch, yaw;
    quaternion_to_rpy(qx, qy, qz, qw, roll, pitch, yaw);

    if (std::abs(roll) > kMaxTiltRad || std::abs(pitch) > kMaxTiltRad)
      return BT::NodeStatus::FAILURE;

    double j5;
    if (!map_yaw_to_j5(yaw, j5))
      return BT::NodeStatus::FAILURE;

    double z_grasp = bz - grasp_depth;
    double z_approach = z_grasp + approach_height;

    setOutput("z_grasp", z_grasp);
    setOutput("z_approach", z_approach);
    setOutput("j5", j5);

    return BT::NodeStatus::SUCCESS;
  }
};

}  // namespace monty_bt
