#ifndef PLANE_SEGMENTATION_HPP
#define PLANE_SEGMENTATION_HPP

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <Eigen/Geometry>
#include <memory>

namespace pose_estimation {

/**
 * @brief Result structure for plane segmentation
 */
struct PlaneSegmentationResult {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cloud;

    PlaneSegmentationResult() {
        remaining_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        planes_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        largest_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }
};

/**
 * @brief Class for segmenting and removing planes from point clouds
 */
class PlaneSegmentation {
public:
    /**
     * @brief Configuration for plane segmentation
     */
    struct Config {
        bool colorize_planes = true;
        size_t min_plane_points = 1000;
        float min_remaining_percent = 0.1f;
        int max_planes = 3;
        float distance_threshold = 0.02f;
        int max_iterations = 1000;
    };

    /**
     * @brief Constructor with logger
     * @param logger ROS2 logger
     * @param config Configuration parameters
     */
    PlaneSegmentation(
        rclcpp::Logger logger,
        const Config& config);

    /**
     * @brief Set the camera to map transform
     * @param transform The transformation matrix
     */
    void setTransform(const Eigen::Matrix4f& transform);

    /**
     * @brief Update configuration parameters
     * @param config New configuration
     */
    void setConfig(const Config& config);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& getConfig() const;

    /**
     * @brief Detect and remove planes from a point cloud
     * @param input_cloud Input point cloud
     * @return Result containing remaining cloud, planes cloud, largest plane cloud
     */
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeMainPlanes(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud);

    /**
     * @brief Get the full segmentation result from the last operation
     * @return PlaneSegmentationResult with all cloud components
     */
    const PlaneSegmentationResult& getLastResult() const;

private:
    /**
     * @brief Check if a plane is a floor plane
     * @param plane_cloud Cloud containing the plane points
     * @param coefficients Plane coefficients
     * @param height_threshold Output parameter for floor height threshold
     * @return True if the plane is the floor, false otherwise
     */
    bool checkFloorPlane(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& plane_cloud,
        const pcl::ModelCoefficients::Ptr& coefficients,
        float& height_threshold);

    /**
     * @brief Filter points that are close to the floor
     * @param cloud Cloud to filter (in-place)
     * @param coefficients Floor plane coefficients
     * @param height_threshold Height threshold for filtering
     */
    void filterFloorClouds(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
        const pcl::ModelCoefficients::Ptr& coefficients,
        const float height_threshold);

    rclcpp::Logger logger_;
    Config config_;
    Eigen::Matrix4f camera_to_map_transform_ = Eigen::Matrix4f::Zero();
    PlaneSegmentationResult last_result_;
    bool debug_time_ = false;
};

} // namespace pose_estimation

#endif // PLANE_SEGMENTATION_HPP