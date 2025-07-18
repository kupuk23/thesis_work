#ifndef CLOUD_PREPROCESS_HPP
#define CLOUD_PREPROCESS_HPP

#include <pose_estimation_pcl/plane_segmentation.hpp>
#include <pose_estimation_pcl/clustering.hpp>


#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <memory>
#include <cfloat> // For FLT_MAX


namespace pose_estimation {

class PlaneSegmentation;
class CloudClustering;

/**
 * @brief Main class for preprocessing point clouds for pose estimation
 * 
 * Coordinates the preprocessing pipeline:
 * 1. Downsampling using a voxel grid
 * 2. Filtering by coordinate boundaries
 * 3. Plane detection and removal (using PlaneSegmentation)
 * 4. Clustering (using CloudClustering)
 */
class PointCloudPreprocess {
public:
    /**
     * @brief Configuration parameters for preprocessing
     */
    struct Config {
        // Pipeline control
        bool enable_plane_removal = true;
        bool enable_clustering = true;
        
        // Voxel grid parameters
        float voxel_size = 0.05f;
        
        // Passthrough filter parameters
        float x_max = 3.0f; 
    };
    
    /**
     * @brief Constructor with logger and optional configuration
     */
    PointCloudPreprocess(
        const Config& config,
        std::shared_ptr<PlaneSegmentation> plane_segmenter,
        std::shared_ptr<CloudClustering> cloud_clusterer,
        bool debug_time = false,
    rclcpp::Logger logger = rclcpp::get_logger("point_cloud_preprocess"));
    
    /**
     * @brief Process a point cloud using the configured pipeline
     * 
     * @param input_cloud Input point cloud
     * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr Processed point cloud
     */
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr process(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
        const float floor_height = -FLT_MAX);
    
    /**
     * @brief Set a new configuration
     */
    void setConfig(const Config& config);
    
    /**
     * @brief Get the current configuration
     */
    const Config& getConfig() const;
    
    /**
     * @brief Get the plane segmenter
     */
    std::shared_ptr<PlaneSegmentation> getPlaneSegmentation() const;
    
    /**
     * @brief Get the cloud clusterer
     */
    std::shared_ptr<CloudClustering> getCloudClustering() const;
    
private:
    // Preprocessing methods
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampleCloud(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud);
        
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr applyPassthroughFilters(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
        const float floor_height);

    // Class members
    Config config_;
    rclcpp::Logger logger_;
    std::shared_ptr<PlaneSegmentation> plane_segmenter_ = nullptr;
    std::shared_ptr<CloudClustering> cloud_clusterer_ = nullptr;
    bool debug_time_ = false;
    // std::shared_ptr<CloudClustering> cloud_clusterer_;
};

} // namespace pose_estimation

#endif // POINT_CLOUD_PREPROCESS_HPP