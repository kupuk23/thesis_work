#ifndef ROS_UTILS_HPP
#define ROS_UTILS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Core>
#include <tf2_ros/buffer.h>

namespace ros_utils {



/**
 * @brief Publish a debug point cloud
 * 
 * @param cloud The point cloud to publish
 * @param cloud_msg The original point cloud message (for header information)
 * @param debug_publisher The publisher to use
 * @param enable_debug Flag to control if debug publishing is enabled
 */
void publish_debug_cloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& debug_publisher,
    bool enable_debug = true);
    

/**
 * @brief Publish an array of floats as a ROS Float32MultiArray message
 * 
 * @param values Array of float values to publish
 * @param array_publisher Publisher to use
 */
void publish_array(const rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr& array_publisher, 
                    const std::vector<float>& values 
                  );

/**
 * @brief Publish an empty pose (identity matrix)
 * 
 * @param publisher Publisher to use
 * @param cloud_msg Original point cloud message
 */

void publish_empty_pose(
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& publisher,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg);

/**
 * @brief Publish registration results
 * 
 * @param transform The final transformation matrix
 * @param tf_buffer TF2 buffer for transformation lookups
 * @param cloud_msg The original point cloud message (for timestamp and frame_id)
 * @param pose_publisher Publisher for the aligned pose
 * @param tf_broadcaster Transform broadcaster
 * @param object_frame The object frame name
 * @param suffix Suffix for the child frame ID
 
 */
void publish_registration_results(
    const Eigen::Matrix4f& transform,
    tf2_ros::Buffer* tf_buffer,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& pose_publisher,
    std::shared_ptr<tf2_ros::TransformBroadcaster>& tf_broadcaster,
    const std::string& object_frame,
    const std::string& suffix
);

/**
 * @brief Convert Eigen transformation matrix to ROS Pose message
 * 
 * @param transform 4x4 transformation matrix 
 * @return geometry_msgs::msg::Pose Converted pose message
 */
geometry_msgs::msg::Pose matrix_to_pose(const Eigen::Matrix4f& transform);

/**
 * @brief Convert ROS Pose message to Eigen transformation matrix
 * 
 * @param pose ROS Pose message
 * @return Eigen::Matrix4f 4x4 transformation matrix
 */
Eigen::Matrix4f pose_to_matrix(const geometry_msgs::msg::Pose& pose);



/**
 * Looks up a transform between two frames using tf2.
 *
 * @param tf_buffer The tf2 buffer to use for lookups
 * @param target_frame The target frame ID
 * @param source_frame The source frame ID
 * @return The transformation as an Eigen::Matrix4f, identity if lookup fails
 */
Eigen::Matrix4f lookup_transformation(
    tf2_ros::Buffer* tf_buffer,
    const std::string& target_frame,
    const std::string& source_frame);


/**
 * @brief Transform object pose from map frame to camera frame
 * 
 * @param pc2_msg PointCloud2 message (used for header info)
 * @param tf_buffer TF2 buffer for transformation lookups
 * @param obj_frame Object frame name (default: "handrail")
 * @param logger Logger for output messages
 * @return Eigen::Matrix4f Transformed object pose as matrix, or identity if transform failed
 */
Eigen::Matrix4f transform_obj_pose(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc2_msg,
    tf2_ros::Buffer& tf_buffer,
    const std::string& obj_frame,
    const rclcpp::Logger& logger);

/**
 * @brief Create a TransformStamped message from a transformation matrix
 * 
 * @param transform The 4x4 transformation matrix
 * @param header_stamp The timestamp for the transform header
 * @param header_frame_id The frame ID for the transform header
 * @param child_frame_id The child frame ID for the transform
 * @return geometry_msgs::msg::TransformStamped The created TransformStamped message
 */
geometry_msgs::msg::TransformStamped create_transform_stamped(
    const Eigen::Matrix4f& transform,
    const rclcpp::Time& header_stamp,
    const std::string& header_frame_id,
    const std::string& child_frame_id);

/**
 * @brief Apply Gaussian noise to a transformation matrix
 * 
 * @param transform The original transformation matrix
 * @param t_std Standard deviation for translation noise
 * @param r_std Standard deviation for rotation noise
 * @return Eigen::Matrix4f The noisy transformation matrix
 */
Eigen::Matrix4f apply_noise_to_transform(
    const Eigen::Matrix4f& transform, 
    float t_std, 
    float r_std);
    
/**
 * @brief Broadcast a transformation matrix to the TF tree
 * 
 * @param broadcaster The TF broadcaster
 * @param transform The 4x4 transformation matrix
 * @param header_stamp The timestamp for the transform header
 * @param header_frame_id The frame ID for the transform header
 * @param child_frame_id The child frame ID for the transform
 */
void broadcast_transform(
    std::shared_ptr<tf2_ros::TransformBroadcaster>& broadcaster,
    const Eigen::Matrix4f& transform,
    const rclcpp::Time& header_stamp,
    const std::string& header_frame_id,
    const std::string& child_frame_id);

/**
 * @brief Apply Gaussian noise to a transformation matrix
 * 
 * @param transform The original transformation matrix
 * @param t_std Standard deviation for translation noise
 * @param r_std Standard deviation for rotation noise
 * @return Eigen::Matrix4f The noisy transformation matrix
 */
Eigen::Matrix4f apply_noise_to_transform(
    const Eigen::Matrix4f& transform, 
    float t_std, 
    float r_std);


} // namespace ros_utils

#endif // ROS_UTILS_HPP