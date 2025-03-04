/*
 * Copyright (C) 2025 [Your Name/Organization]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

 #ifndef GZ_SIM_SYSTEMS_SixDOFVelocityControl_HH_
 #define GZ_SIM_SYSTEMS_SixDOFVelocityControl_HH_
 
 #include <memory>
 #include <gz/sim/System.hh>
 
 namespace gz
 {
 namespace sim
 {
 // Inline bracket to help doxygen filtering.
 inline namespace GZ_SIM_VERSION_NAMESPACE {
 namespace systems
 {
   // Forward declaration
   class SixDOFVelocityControlPrivate;
 
   /// \brief A plugin that applies linear and angular velocities to a link
   /// for simulating movement in a microgravity environment.
   /// This plugin receives velocity commands via a ROS topic and applies them
   /// in the link's local frame.
   ///
   /// ## System Parameters
   ///
   /// - `<topic_namespace>`: Namespace for the ROS topics. Default is the model name.
   /// - `<link_name>`: Name of the link to apply velocities to. Required.
   /// - `<cmd_vel_topic>`: ROS topic for velocity commands. Default: "/cmd_vel"
   /// - `<linear_velocity_scale>`: Scaling factor for linear velocity. Default: 1.0
   /// - `<angular_velocity_scale>`: Scaling factor for angular velocity. Default: 1.0
   ///
   /// ## Example
   ///
   /// ```
   /// <plugin
   ///   filename="libSixDOFVelocityControl.so"
   ///   name="gz::sim::systems::SixDOFVelocityControl">
   ///   <topic_namespace>spacecraft</topic_namespace>
   ///   <link_name>spacecraft_body</link_name>
   ///   <cmd_vel_topic>/cmd_vel</cmd_vel_topic>
   ///   <linear_velocity_scale>1.0</linear_velocity_scale>
   ///   <angular_velocity_scale>1.0</angular_velocity_scale>
   /// </plugin>
   /// ```
   class SixDOFVelocityControl
       : public System,
         public ISystemConfigure,
         public ISystemPreUpdate
   {
     /// \brief Constructor
     public: SixDOFVelocityControl();
 
     /// \brief Destructor
     public: ~SixDOFVelocityControl() override = default;
 
     // Documentation inherited
     public: void Configure(const Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           EntityComponentManager &_ecm,
                           EventManager &_eventMgr) override;
 
     // Documentation inherited
     public: void PreUpdate(
                 const UpdateInfo &_info,
                 EntityComponentManager &_ecm) override;
 
     /// \brief Private data pointer
     private: std::unique_ptr<SixDOFVelocityControlPrivate> dataPtr;
   };
 }
 }
 }
 }
 
 #endif