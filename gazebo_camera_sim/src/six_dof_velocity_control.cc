// #include <gz/plugin/Register.hh>
// #include <gz/sim/System.hh>
// #include <gz/sim/Model.hh>
// #include <gz/transport/Node.hh>
// #include <gz/msgs/Utility.hh>

// #include <rclcpp/rclcpp.hpp>
// #include <geometry_msgs/msg/twist.hpp>

// class SixDOFVelocityControl : public gz::sim::System,
//                               public gz::sim::ISystemConfigure,
//                               public gz::sim::ISystemPreUpdate
// {
// public:
//     void Configure(const gz::sim::Entity &_entity,
//                    const std::shared_ptr<const sdf::Element> &_sdf,
//                    gz::sim::EntityComponentManager &_ecm,
//                    gz::sim::EventManager &_eventMgr) override {
//         // Initialize ROS node
//         if (!rclcpp::ok()) {
//             rclcpp::init(0, nullptr);
//         }
//         this->rosNode = std::make_shared<rclcpp::Node>("sixdof_velocity_control");
//         this->subscription = this->rosNode->create_subscription<geometry_msgs::msg::Twist>(
//             "/cmd_vel", 10, std::bind(&SixDOFVelocityControl::OnVelMsg, this, std::placeholders::_1));

//         this->model = gz::sim::Model(_entity, _ecm);
//         if (!this->model.Valid()) {
//             gzerr << "SixDOFVelocityControl plugin should be attached to a model entity. Failed to initialize." << std::endl;
//             return;
//         }
//     }

//     void PreUpdate(const gz::sim::UpdateInfo &_info,
//                    gz::sim::EntityComponentManager &_ecm) override {
//         // Apply the stored velocities to the model
//         if (!this->linearVelocity.empty() && !this->angularVelocity.empty()) {
//             this->model.SetWorldLinearVelocity(gz::math::Vector3d(
//                 this->linearVelocity[0], this->linearVelocity[1], this->linearVelocity[2]), _ecm);
//             this->model.SetWorldAngularVelocity(gz::math::Vector3d(
//                 this->angularVelocity[0], this->angularVelocity[1], this->angularVelocity[2]), _ecm);
//         }
//     }

// private:
//     void OnVelMsg(const geometry_msgs::msg::Twist::SharedPtr msg) {
//         // Convert ROS 2 Twist message to Gazebo velocities
//         this->linearVelocity = {msg->linear.x, msg->linear.y, msg->linear.z};
//         this->angularVelocity = {msg->angular.x, msg->angular.y, msg->angular.z};
//     }

//     rclcpp::Node::SharedPtr rosNode;
//     rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription;
//     gz::sim::Model model;
//     std::vector<double> linearVelocity;
//     std::vector<double> angularVelocity;
// };

// GZ_ADD_PLUGIN(SixDOFVelocityControl,
//               gz::sim::System,
//               SixDOFVelocityControl::ISystemConfigure,
//               SixDOFVelocityControl::ISystemPreUpdate)
