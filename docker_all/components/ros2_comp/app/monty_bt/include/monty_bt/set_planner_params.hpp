#pragma once

#include <string>
#include <behaviortree_ros2/bt_service_node.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/parameter_value.hpp>

namespace monty_bt
{

class SetPlannerParams
    : public BT::RosServiceNode<rcl_interfaces::srv::SetParameters>
{
public:
  SetPlannerParams(const std::string& name, const BT::NodeConfig& conf,
                   const BT::RosNodeParams& params)
      : RosServiceNode(name, conf, params)
  {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
        BT::InputPort<double>("target_x"),
        BT::InputPort<double>("target_y"),
        BT::InputPort<double>("target_z"),
        BT::InputPort<double>("target_pitch"),
        BT::InputPort<double>("target_roll"),
        BT::InputPort<double>("target_grip"),
        BT::InputPort<bool>("execute"),
    });
  }

  bool setRequest(Request::SharedPtr& request) override
  {
    auto add_double = [&](const char* port_name, const char* param_name) {
      if (auto val = getInput<double>(port_name))
      {
        rcl_interfaces::msg::Parameter p;
        p.name = param_name;
        p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
        p.value.double_value = val.value();
        request->parameters.push_back(p);
      }
    };

    add_double("target_x", "target_x");
    add_double("target_y", "target_y");
    add_double("target_z", "target_z");
    add_double("target_pitch", "target_pitch");
    add_double("target_roll", "target_roll");
    add_double("target_grip", "target_grip");

    if (auto val = getInput<bool>("execute"))
    {
      rcl_interfaces::msg::Parameter p;
      p.name = "execute";
      p.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
      p.value.bool_value = val.value();
      request->parameters.push_back(p);
    }

    if (request->parameters.empty())
    {
      RCLCPP_WARN(logger(), "SetPlannerParams [%s]: no ports wired", name().c_str());
      return false;
    }
    return true;
  }

  BT::NodeStatus onResponseReceived(const Response::SharedPtr& response) override
  {
    for (const auto& r : response->results)
    {
      if (!r.successful)
      {
        RCLCPP_ERROR(logger(), "SetPlannerParams: param set failed: %s",
                     r.reason.c_str());
        return BT::NodeStatus::FAILURE;
      }
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override
  {
    RCLCPP_ERROR(logger(), "SetPlannerParams [%s]: %s", name().c_str(),
                 BT::toStr(error));
    return BT::NodeStatus::FAILURE;
  }
};

}  // namespace monty_bt
