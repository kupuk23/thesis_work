#ifndef PLANE_SEGMENTATION_HPP
#define PLANE_SEGMENTATION_HPP

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <Eigen/Geometry>
#include <memory>
#include <cfloat> // For FLT_MAX

namespace pose_estimation
{
    enum class PlaneType
    {
        FLOOR,
        CEILING,
        WALL_X, // Wall aligned with X axis
        WALL_Y, // Wall aligned with Y axis
        UNKNOWN
    };

    /**
     * @brief Result structure for plane segmentation
     */
    struct PlaneSegmentationResult
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cloud;

        PlaneSegmentationResult()
        {
            remaining_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
            planes_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
            largest_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        }
    };

    /**
     * @brief Class for segmenting and removing planes from point clouds
     */
    class PlaneSegmentation
    {
    public:
        /**
         * @brief Configuration for plane segmentation
         */
        struct Config
        {
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
            const Config &config,
            rclcpp::Logger logger = rclcpp::get_logger("plane_segmentation"));

        /**
         * @brief Set the camera to map transform
         * @param transform The transformation matrix
         */
        void setTransform(const Eigen::Matrix4f &transform);

        /**
         * @brief Update configuration parameters
         * @param config New configuration
         */
        void setConfig(const Config &config);

        /**
         * @brief Get current configuration
         * @return Current configuration
         */
        const Config &getConfig() const;

        /**
         * @brief Detect and remove planes from a point cloud
         * @param input_cloud Input point cloud
         * @param axis Axis to consider for plane detection (default is Y axis)
         * @param eps_deg Maximum angle deviation in degrees for plane detection
         * @return Result containing remaining cloud, planes cloud, largest plane cloud
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeMainPlanes(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
            const Eigen::Vector3f &axis = Eigen::Vector3f(0, 1, 0), // Default to Y axis (walls in front of camera)
            float eps_deg = 5.0f,
            bool measuring_dist = false);

        /**
         * @brief Get the full segmentation result from the last operation
         * @return PlaneSegmentationResult with all cloud components
         */
        const PlaneSegmentationResult &getLastResult() const;

        /**
         * @brief Measure the distance to the floor plane
         * @param input_cloud Input point cloud
         * @param eps_deg Angular tolerance in degrees
         * @return Estimated distance to the floor plane
         */
        float measureFloorDist(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
                               float eps_deg);

    private:
        /**
         * @brief Check if a plane is a floor plane
         * @param plane_cloud Cloud containing the plane points
         * @param coefficients Plane coefficients
         * @param dist_threshold Output parameter for floor height threshold
         * @param offset Offset to apply to the height threshold
         * @return True if the plane is the floor, false otherwise
         */
        bool getPlaneDistance(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud,
            const pcl::ModelCoefficients::Ptr &coefficients,
            float &dist_threshold,
            const Eigen::Vector3f &plane_axis,
            float offset = 0.15f);

        /**
         * @brief Filter points that are close to the floor
         * @param cloud Cloud to filter (in-place)
         * @param plane_axis axis to filter along
         * @param plane_direction Direction of the plane normal
         * @param dist_threshold Height threshold for filtering
         */
        void filterCloudsByPlane(
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            const std::string &plane_axis,
            const std::string &plane_direction,
            const float dist_threshold);

        rclcpp::Logger logger_;
        Config config_;
        Eigen::Matrix4f camera_to_map_transform_ = Eigen::Matrix4f::Zero();
        float floor_height_ = -FLT_MAX; // Height of the floor plane
        PlaneSegmentationResult last_result_;
        bool debug_time_ = false;
    };

} // namespace pose_estimation

#endif // PLANE_SEGMENTATION_HPP