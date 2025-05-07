#ifndef PLANE_SEGMENTATION_HPP
#define PLANE_SEGMENTATION_HPP

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <vector>
#include <array>
#include <memory>

namespace pose_estimation {

/**
 * @brief Class for detecting and removing planes from point clouds
 */
class PlaneSegmentation {
public:
    /**
     * @brief Configuration parameters for plane segmentation
     */
    struct Config {
        double plane_distance_threshold = 0.02;
        int max_plane_iterations = 100;
        int min_plane_points = 800;
        int max_planes = 3;
        float min_remaining_percent = 0.2f;
        bool colorize_planes = true;
    };
    
    /**
     * @brief Result structure for plane segmentation
     */
    struct Result {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cloud;
        
        Result() :
            remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()),
            planes_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()),
            largest_plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>()) {}
    };
    
    /**
     * @brief Constructor with logger and optional configuration
     */
    PlaneSegmentation(rclcpp::Logger logger, const Config& config = Config());
    
    /**
     * @brief Detect and remove planes from a point cloud
     * 
     * @param input_cloud Input point cloud
     * @return Result Segmentation result containing remaining points and plane points
     */
    Result detectAndRemovePlanes(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud);
    
    /**
     * @brief Get the last segmentation result
     */
    const Result& getLastResult() const;
    
    /**
     * @brief Set a new configuration
     */
    void setConfig(const Config& config);
    
    /**
     * @brief Get the current configuration
     */
    const Config& getConfig() const;
    
private:
    // Predefined colors for visualization
    static const std::vector<std::array<uint8_t, 3>> DEFAULT_COLORS;
    
    // Class members
    Config config_;
    rclcpp::Logger logger_;
    Result last_result_;
};

} // namespace pose_estimation

#endif // PLANE_SEGMENTATION_HPP