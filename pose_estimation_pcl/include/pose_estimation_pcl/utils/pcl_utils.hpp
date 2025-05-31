#ifndef PCL_UTILS_HPP
#define PCL_UTILS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/ModelCoefficients.h>
#include <Eigen/Core>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>


// Add to pcl_utils.hpp
namespace pcl_utils {

// Define color constants
const std::vector<std::array<uint8_t, 3>> DEFAULT_COLORS = {
    {255, 0, 0},    // Red
    {0, 255, 0},    // Green
    {0, 0, 255},    // Blue
    {255, 255, 0},  // Yellow
    {255, 0, 255},  // Magenta
    {0, 255, 255},  // Cyan
    {255, 128, 0},  // Orange
    {128, 0, 255},  // Purple
    {0, 128, 255},  // Light Blue
    {255, 0, 128}   // Pink
};



/**
 * @brief load a point cloud from a file
 * @param filename Path to the file
 * @param logger Logger for output messages
 * @return Point cloud loaded from the file
 * **/
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadCloudFromFile(
    const std::string& filename,
    const rclcpp::Logger& logger = rclcpp::get_logger("load_model_pcd"));

/**
 * @brief Visualize point cloud with normals for debugging
 * 
 * @param cloud Input point cloud
 * @param normals Computed surface normals
 * @param normal_length Display length of normal vectors (default: 0.02)
 * @param point_size Size of points in visualization (default: 3)
 * @param window_name Name of the visualization window (default: "Cloud with Normals")
 * @param blocking If true, blocks until window is closed (default: true)
 */
void visualizeNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float normal_length = 0.02,
    int point_size = 3,
    const std::string& window_name = "Cloud with Normals",
    bool blocking = true);


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
 * @brief Load a model PCD file
 * 
 * @param filename PCD file path
 * @param logger Logger for output messages
 * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr Loaded point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadModelPCD(
    const std::string& filename,
    const rclcpp::Logger& logger = rclcpp::get_logger("load_model_pcd"));


} // namespace pcl_utils



#endif // PCL_UTILS_HPP