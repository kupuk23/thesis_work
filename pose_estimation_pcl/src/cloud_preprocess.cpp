#include "pose_estimation_pcl/cloud_preprocess.hpp"
#include "pose_estimation_pcl/plane_segmentation.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <cfloat>  // For FLT_MAX

namespace pose_estimation {

// Constructor implementation
PointCloudPreprocess::PointCloudPreprocess(
    
    const Config& config,
    std::shared_ptr<PlaneSegmentation> plane_segmenter,
    std::shared_ptr<CloudClustering> cloud_clusterer,
    bool debug_time,
    rclcpp::Logger logger)
    : config_(config), plane_segmenter_(plane_segmenter), cloud_clusterer_(cloud_clusterer) ,debug_time_(debug_time), logger_(logger)
{
    RCLCPP_INFO(logger_, "Point cloud preprocessor initialized");
}

// Process a point cloud using the configured pipeline
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPreprocess::process(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
{
    if (!input_cloud || input_cloud->empty()) {
        RCLCPP_WARN(logger_, "Input cloud is empty or null");
        return input_cloud;
    }

    RCLCPP_DEBUG(logger_, "Processing point cloud with %zu points", input_cloud->size());
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    // Step 1: Downsample the cloud
    auto downsampled_cloud = downsampleCloud(input_cloud);
    
    // Step 2: Apply passthrough filters
    auto filtered_cloud = applyPassthroughFilters(downsampled_cloud);

    if (filtered_cloud->empty()) {
        RCLCPP_WARN(logger_, "Downsampled cloud is empty after passthrough filter");
        filtered_cloud = downsampled_cloud;  // Fallback to downsampled cloud
    }
    
    // // Step 3: Plane detection and removal (if enabled and segmenter exists)

    // define filtered_cloud
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*ds_cloud));
    
    // if (plane_segmenter_){
    //     filtered_cloud = plane_segmenter_->removeMainPlanes(filtered_cloud);
    // } else {    
    //     RCLCPP_DEBUG(logger_, "Plane segmentation skipped - segmenter not initialized");
    // }

    
    // // // Step 4: Clustering (if enabled and clusterer exists)
    // if (config_.enable_clustering && cloud_clusterer_) {
    //     // This would be implemented with your clustering class
    //     filtered_cloud = cloud_clusterer_->extractMainCluster(filtered_cloud);
    //     RCLCPP_DEBUG(logger_, "Clustering skipped - clusterer not initialized");
    // }
    
    RCLCPP_DEBUG(logger_, "Processing complete, output cloud has %zu points", filtered_cloud->size());
    
    // end time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    if (debug_time_) {
        RCLCPP_INFO(logger_, "Point cloud preprocessing executed in %.3f seconds", execution_time);
    }
    return filtered_cloud;
}

// Set a new configuration
void PointCloudPreprocess::setConfig(const Config& config) {
    config_ = config;
    RCLCPP_INFO(logger_, "Preprocessor configuration updated");
}

// Get the current configuration
const PointCloudPreprocess::Config& PointCloudPreprocess::getConfig() const {
    return config_;
}

// Get the plane segmenter
std::shared_ptr<PlaneSegmentation> PointCloudPreprocess::getPlaneSegmentation() const {
    return plane_segmenter_;
}

// // Get the cloud clusterer
// std::shared_ptr<CloudClustering> PointCloudPreprocess::getCloudClustering() const {
//     return cloud_clusterer_;
// }

// Private methods

// Downsample cloud using voxel grid
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPreprocess::downsampleCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    voxel_grid.setInputCloud(input_cloud);
    voxel_grid.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
    voxel_grid.filter(*output_cloud);
    
    RCLCPP_DEBUG(logger_, "Downsampled cloud from %zu to %zu points", 
                 input_cloud->size(), output_cloud->size());
    
    return output_cloud;
}

// Apply passthrough filters for coordinate boundaries
pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPreprocess::applyPassthroughFilters(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    

    // Filter by X coordinate (forward distance)
    pcl::PassThrough<pcl::PointXYZRGB> pass_x;
    pass_x.setInputCloud(input_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(-FLT_MAX, config_.x_max);  // Keep points within x_max
    pass_x.filter(*filtered_cloud);
    
    RCLCPP_DEBUG(logger_, "Passthrough filter reduced cloud from %zu to %zu points", 
                 input_cloud->size(), filtered_cloud->size());
    
    return filtered_cloud;
}

} // namespace pose_estimation