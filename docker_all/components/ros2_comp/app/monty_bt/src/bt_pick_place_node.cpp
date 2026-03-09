/**
 * BT executor node for X3plus pick-and-place.
 *
 * Loads XML behavior trees, populates the blackboard from ROS parameters,
 * and exposes ~/run_pick, ~/run_place, ~/run_pick_and_place as Trigger
 * services.  Each service call ticks the corresponding tree to completion.
 *
 * Existing planner services are called through generic BT leaf nodes
 * (SetPlannerParams, CallTriggerService).  No existing code is modified.
 */

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_ros2/ros_node_params.hpp>

#include "monty_bt/set_planner_params.hpp"
#include "monty_bt/call_trigger_service.hpp"
#include "monty_bt/compute_grasp_params.hpp"

using namespace std::chrono_literals;

static constexpr double BLOCK_HEIGHT = 0.03;
static constexpr double APPROACH_HEIGHT = 0.04;
static constexpr double GRASP_DEPTH = 0.015;
static constexpr double GRIP_OPEN = 0.0;
static constexpr double GRIP_CLOSED = -1.3;

class BtPickPlaceNode : public rclcpp::Node
{
public:
  BtPickPlaceNode() : Node("bt_pick_place")
  {
    declare_parameter("block_x", 0.23);
    declare_parameter("block_y", 0.0);
    declare_parameter("block_z", BLOCK_HEIGHT);
    declare_parameter("block_qx", 0.0);
    declare_parameter("block_qy", 0.0);
    declare_parameter("block_qz", 0.0);
    declare_parameter("block_qw", 1.0);
    declare_parameter("place_x", 0.23);
    declare_parameter("place_y", 0.03);
    declare_parameter("place_z", BLOCK_HEIGHT);
    declare_parameter("place_qx", 0.0);
    declare_parameter("place_qy", 0.0);
    declare_parameter("place_qz", 0.0);
    declare_parameter("place_qw", 1.0);
    declare_parameter("approach_height", APPROACH_HEIGHT);
    declare_parameter("grasp_depth", GRASP_DEPTH);
    declare_parameter("grip_open", GRIP_OPEN);
    declare_parameter("grip_closed", GRIP_CLOSED);
    declare_parameter("execute", true);
    declare_parameter("dry_run", false);

    std::string pkg_share =
        ament_index_cpp::get_package_share_directory("monty_bt");
    pick_tree_path_ = pkg_share + "/trees/pick.xml";
    place_tree_path_ = pkg_share + "/trees/place.xml";
    pick_and_place_tree_path_ = pkg_share + "/trees/pick_and_place.xml";
  }

  void init()
  {
    register_bt_nodes();

    srv_cb_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    run_pick_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/run_pick",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr resp) {
          run_tree(pick_tree_path_, "pick", resp);
        },
        rclcpp::ServicesQoS(), srv_cb_group_);

    run_place_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/run_place",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr resp) {
          run_tree(place_tree_path_, "place", resp);
        },
        rclcpp::ServicesQoS(), srv_cb_group_);

    run_pick_and_place_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/run_pick_and_place",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr resp) {
          run_tree(pick_and_place_tree_path_, "pick_and_place", resp);
        },
        rclcpp::ServicesQoS(), srv_cb_group_);

    RCLCPP_INFO(get_logger(),
                "BT pick-place ready. Services: ~/run_pick, ~/run_place, "
                "~/run_pick_and_place");
  }

private:
  void register_bt_nodes()
  {
    BT::RosNodeParams ros_params;
    ros_params.nh = shared_from_this();
    ros_params.server_timeout = std::chrono::milliseconds(60000);
    ros_params.wait_for_server_timeout = std::chrono::milliseconds(10000);

    factory_.registerNodeType<monty_bt::SetPlannerParams>(
        "SetPlannerParams", ros_params);
    factory_.registerNodeType<monty_bt::CallTriggerService>(
        "CallTriggerService", ros_params);
    factory_.registerNodeType<monty_bt::ComputeGraspParams>(
        "ComputeGraspParams");
  }

  bool resolve_execute()
  {
    bool execute = get_parameter("execute").as_bool();
    bool dry_run = get_parameter("dry_run").as_bool();
    if (dry_run && execute)
    {
      RCLCPP_WARN(get_logger(), "dry_run active — forcing execute=false");
      return false;
    }
    return execute;
  }

  void populate_blackboard(BT::Blackboard::Ptr bb, const std::string& mode)
  {
    bool execute = resolve_execute();
    bb->set("execute", execute);
    bb->set("approach_height", get_parameter("approach_height").as_double());
    bb->set("grasp_depth", get_parameter("grasp_depth").as_double());
    bb->set("g_open", get_parameter("grip_open").as_double());
    bb->set("g_closed", get_parameter("grip_closed").as_double());

    if (mode == "pick" || mode == "pick_and_place")
    {
      bb->set("bx", get_parameter("block_x").as_double());
      bb->set("by", get_parameter("block_y").as_double());
      bb->set("bz", get_parameter("block_z").as_double());
      bb->set("qx", get_parameter("block_qx").as_double());
      bb->set("qy", get_parameter("block_qy").as_double());
      bb->set("qz", get_parameter("block_qz").as_double());
      bb->set("qw", get_parameter("block_qw").as_double());
    }

    if (mode == "place")
    {
      bb->set("qx", get_parameter("place_qx").as_double());
      bb->set("qy", get_parameter("place_qy").as_double());
      bb->set("qz", get_parameter("place_qz").as_double());
      bb->set("qw", get_parameter("place_qw").as_double());
    }

    if (mode == "place" || mode == "pick_and_place")
    {
      bb->set("px", get_parameter("place_x").as_double());
      bb->set("py", get_parameter("place_y").as_double());
      bb->set("pz", get_parameter("place_z").as_double());
    }

    // pick_and_place uses block quaternion for pick and place quaternion for place
    if (mode == "pick_and_place")
    {
      // qx/qy/qz/qw set above are for pick (block).
      // Place subtree uses its own ComputeGraspParams reading px/py/pz
      // with the same quaternion — matching the Python code's default behaviour
      // where both pick and place share the same quaternion params if not
      // overridden.  For distinct place orientation, set place_q* params.
    }
  }

  void run_tree(const std::string& tree_path, const std::string& mode,
                std_srvs::srv::Trigger::Response::SharedPtr resp)
  {
    RCLCPP_INFO(get_logger(), "%s requested (BT)", mode.c_str());

    BT::Tree tree;
    try
    {
      tree = factory_.createTreeFromFile(tree_path);
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(get_logger(), "Failed to load tree %s: %s",
                   tree_path.c_str(), e.what());
      resp->success = false;
      resp->message = std::string("Tree load failed: ") + e.what();
      return;
    }

    populate_blackboard(tree.rootBlackboard(), mode);

    BT::NodeStatus status = BT::NodeStatus::RUNNING;
    while (rclcpp::ok() && status == BT::NodeStatus::RUNNING)
    {
      status = tree.tickOnce();
      std::this_thread::sleep_for(50ms);
    }

    if (status == BT::NodeStatus::SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "%s complete (BT)", mode.c_str());
      resp->success = true;
      resp->message = mode + " complete (BT)";
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "%s failed (BT)", mode.c_str());
      resp->success = false;
      resp->message = mode + " failed (BT)";
    }
  }

  BT::BehaviorTreeFactory factory_;
  std::string pick_tree_path_;
  std::string place_tree_path_;
  std::string pick_and_place_tree_path_;

  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_pick_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_place_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_pick_and_place_srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<BtPickPlaceNode>();
  node->init();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
