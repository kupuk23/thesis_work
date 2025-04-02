#ifndef PCL_UTILS_HPP
#define PCL_UTILS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <tf2_ros/buffer.h>

namespace pcl_utils {

/**
 * @brief Convert ROS PointCloud2 message to PCL cloud with RGB data
 * 
 * @param cloud_msg Input ROS PointCloud2 message
 * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr Converted PCL point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertPointCloud2ToPCL(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg);

/**
 * @brief Save point cloud to PCD file for debugging
 * 
 * @param cloud Point cloud to save
 * @param filename Output file path
 * @param logger Logger for output messages
 */
void saveToPCD(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
    const std::string& filename,
    const rclcpp::Logger& logger);

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
 * @brief Load a model PCD file
 * 
 * @param filename PCD file path
 * @param logger Logger for output messages
 * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr Loaded point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadModelPCD(
    const std::string& filename,
    const rclcpp::Logger& logger);

/**
 * Convert a 4x4 transformation matrix to a ROS Pose.
 * 
 * @param matrix 4x4 transformation matrix (Eigen::Matrix4d)
 * @return ROS Pose message
 */
geometry_msgs::msg::Pose matrixToPose(const Eigen::Matrix4d& matrix);


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

} // namespace pcl_utils


#endif // PCL_UTILS_HPP