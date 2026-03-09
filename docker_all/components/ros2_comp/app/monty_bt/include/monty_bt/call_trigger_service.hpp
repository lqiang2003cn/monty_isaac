#pragma once

#include <string>
#include <behaviortree_ros2/bt_service_node.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace monty_bt
{

class CallTriggerService
    : public BT::RosServiceNode<std_srvs::srv::Trigger>
{
public:
  CallTriggerService(const std::string& name, const BT::NodeConfig& conf,
                     const BT::RosNodeParams& params)
      : RosServiceNode(name, conf, params)
  {}

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({});
  }

  bool setRequest(Request::SharedPtr& /*request*/) override
  {
    return true;
  }

  BT::NodeStatus onResponseReceived(const Response::SharedPtr& response) override
  {
    if (!response->success)
    {
      RCLCPP_ERROR(logger(), "CallTriggerService [%s]: %s", name().c_str(),
                   response->message.c_str());
      return BT::NodeStatus::FAILURE;
    }
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override
  {
    RCLCPP_ERROR(logger(), "CallTriggerService [%s]: %s", name().c_str(),
                 BT::toStr(error));
    return BT::NodeStatus::FAILURE;
  }
};

}  // namespace monty_bt
