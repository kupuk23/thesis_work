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

// Structure to hold clustering results
struct ClusteringResult {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> individual_clusters;
    
    ClusteringResult() 
        : colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()) 
    {}
};

// Structure to hold histogram matching results
struct HistogramMatchingResult {
    std::vector<float> cluster_similarities;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr best_matching_cluster;
    
    HistogramMatchingResult() 
        : best_matching_cluster(nullptr) {}
};

// Structure to hold plane segmentation results
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

// Structure to hold FPFH features and their average
struct ClusterFeatures {
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features; // local features
    pcl::FPFHSignature33 average_fpfh; // global average features
    
    ClusterFeatures() : fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>()) {}
};

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
 * @brief Detect and remove planes from point cloud
 * 
 * @param input_cloud Input point cloud
 * @param logger ROS logger for output
 * @param colorize_planes Whether to color the planes
 * @param min_plane_points Minimum points to consider a plane
 * @param min_remaining_percent Minimum percentage of points to remain
 * @param max_planes Maximum number of planes to extract
 * @param dist_threshold Distance threshold for plane fitting
 * @param max_iterations Maximum iterations for plane fitting
 * @return PlaneSegmentationResult Segmentation result with planes and remaining points
 */
PlaneSegmentationResult detect_and_remove_planes(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    bool colorize_planes = true,
    size_t min_plane_points = 800,
    float min_remaining_percent = 0.2,
    int max_planes = 3,
    float dist_threshold = 0.02,
    int max_iterations = 100);

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
    double cluster_tolerance = 0.02,
    int min_cluster_size = 100,
    int max_cluster_size = 25000,
    const rclcpp::Logger& logger = rclcpp::get_logger("cluster_point_cloud"));

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

/**
 * @brief Compute FPFH features for each cluster
 * 
 * @param clusters Vector of point clouds, one for each cluster
 * @param normal_radius Radius for normal estimation
 * @param feature_radius Radius for FPFH feature computation
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @param visualize_normals Whether to visualize normals
 * @param logger Logger for output messages
 * @return std::vector<ClusterFeatures> Vector of features for each cluster
 */
std::vector<ClusterFeatures> computeFPFHFeatures(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters,
    float normal_radius = 0.03,
    float feature_radius = 0.5,
    int num_threads = 0,
    bool visualize_normals = false,
    const rclcpp::Logger& logger = rclcpp::get_logger("fpfh_computation"));

/**
 * @brief Find the best matching cluster to the model using histogram matching
 * 
 * This function compares the average FPFH histograms between the model and scene clusters
 * to identify which cluster most likely contains the target object.
 * 
 * @param model_features FPFH features of the model
 * @param cluster_features Vector of FPFH features for each cluster
 * @param similarity_threshold Minimum similarity score to consider a match valid
 * @param logger ROS logger for output messages
 * @return int Index of the best matching cluster or -1 if no match found
 */
HistogramMatchingResult findBestClusterByHistogram(
    const ClusterFeatures& model_features,
    const std::vector<ClusterFeatures>& cluster_features,
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cluster_clouds,
    float similarity_threshold = 0.7,
    const rclcpp::Logger& logger = rclcpp::get_logger("histogram_matcher"));

} // namespace pcl_utils



#endif // PCL_UTILS_HPP