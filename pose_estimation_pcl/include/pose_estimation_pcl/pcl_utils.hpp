#ifndef PCL_UTILS_HPP
#define PCL_UTILS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

struct ClusteringResult {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> individual_clusters;
    
    ClusteringResult() 
        : colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()) 
    {}
};

struct PlaneSegmentationResult {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cloud;
    
    PlaneSegmentationResult()
        : remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()),
          planes_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()),
          largest_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>())
    {}
};

/**
 * @brief Preprocess a point cloud by downsampling and filtering
 * 
 * @param input_cloud Input point cloud
 * @param voxel_size Voxel size for downsampling
 * @param logger ROS logger for output messages
 * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr Preprocessed cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr preprocess_pointcloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    double voxel_size,
    const rclcpp::Logger& logger);

/**
 * @brief Detect and remove planes from point cloud
 * 
 * @param input_cloud Input point cloud
 * @param logger ROS logger for output
 * @param colorize_planes Whether to color the planes
 * @param min_plane_points Minimum points to consider a plane
 * @param min_remaining_percent Minimum percentage of points to remain
 * @param max_planes Maximum number of planes to extract
 * @return PlaneSegmentationResult Segmentation result with planes and remaining points
 */
PlaneSegmentationResult detect_and_remove_planes(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    bool colorize_planes = true,
    size_t min_plane_points = 800,
    float min_remaining_percent = 0.2,
    int max_planes = 3);

/**
 * @brief Cluster a point cloud into separate objects
 * 
 * @param input_cloud Input point cloud
 * @param logger ROS logger for output
 * @param cluster_tolerance Distance tolerance for clustering
 * @param min_cluster_size Minimum points per cluster
 * @param max_cluster_size Maximum points per cluster
 * @return ClusteringResult Clustered point cloud
 */
ClusteringResult cluster_point_cloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    double cluster_tolerance = 0.02,
    int min_cluster_size = 100,
    int max_cluster_size = 25000);

/**
 * @brief Run GICP registration
 * 
 * @param source_cloud Source point cloud
 * @param target_cloud Target point cloud
 * @param initial_transform Initial transformation
 * @param result_transform Output transformation
 * @param fitness_score Output fitness score
 * @return bool True if registration converged
 */
bool runGICP(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,
    const Eigen::Matrix4f& initial_transform,
    Eigen::Matrix4f& result_transform,
    float& fitness_score);


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
    const rclcpp::Logger& logger);





} // namespace pcl_utils



#endif // PCL_UTILS_HPP