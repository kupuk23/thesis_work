#ifndef GO_ICP_WRAPPER_HPP
#define GO_ICP_WRAPPER_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <chrono>
#include <random>

// Go-ICP includes
#include "go_icp/jly_goicp.h"

namespace go_icp {

/**
 * @brief A wrapper class for Go-ICP algorithm
 */
class GoICPWrapper {
public:
    GoICPWrapper();
    ~GoICPWrapper() = default;

    /**
     * @brief Register a model point cloud to a scene point cloud
     * 
     * @param model_cloud The model point cloud (source)
     * @param scene_cloud The scene point cloud (target)
     * @param max_scene_points Maximum number of scene points to use (for performance)
     * @param debug Enable debug output
     * @return Eigen::Matrix4f The registration transformation
     */
    Eigen::Matrix4f registerPointClouds(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& model_cloud,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scene_cloud,
        int max_scene_points = 3000,
        bool debug = false);

    /**
     * @brief Get the last registration error
     * 
     * @return float Registration error
     */
    float getLastError() const { return last_error_; }

    /**
     * @brief Get the last registration time
     * 
     * @return float Registration time in seconds
     */
    float getLastRegistrationTime() const { return last_registration_time_; }

    /**
     * @brief Get the last normalization scale
     * 
     * @return float Normalization scale
     */
    float getLastNormalizationScale() const { return last_normalization_scale_; }

private:
    /**
     * @brief Convert RGB point cloud to XYZ
     * 
     * @param rgb_cloud Input RGB point cloud
     * @return pcl::PointCloud<pcl::PointXYZ>::Ptr Output XYZ point cloud
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr convertRGBtoXYZ(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud);

    /**
     * @brief Downsample a point cloud to a maximum number of points
     * 
     * @param cloud Input point cloud
     * @param max_points Maximum number of points
     * @return pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampled point cloud
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        int max_points);

    /**
     * @brief Normalize point clouds to fit in [-1,1]³ cube
     * 
     * @param data_pc data point cloud
     * @param model_pc model point cloud
     * @param data_normalized Output normalized data
     * @param model_normalized Output normalized model
     * @param data_centroid Output source centroid
     * @param model_centroid Output target centroid
     * @param max_scale Output normalization scale
     */
    void normalizePointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& data_pc,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& model_pc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& data_normalized,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& model_normalized,
        Eigen::Vector3f& data_centroid,
        Eigen::Vector3f& model_centroid,
        float& max_scale);

    /**
     * @brief Denormalize a transformation
     * 
     * @param transform Normalized transformation
     * @param data_centroid data centroid
     * @param model_centroid model centroid
     * @param scale Normalization scale
     * @return Eigen::Matrix4f Denormalized transformation
     */
    Eigen::Matrix4f denormalizeTransformation(
        const Eigen::Matrix4f& transform,
        const Eigen::Vector3f& data_centroid,
        const Eigen::Vector3f& model_centroid,
        float scale);

    /**
     * @brief Convert PCL point cloud to Point3D array for Go-ICP
     * 
     * @param cloud Input point cloud
     * @return std::vector<POINT3D> Output Point3D array
     */
    std::vector<POINT3D> convertToPoint3DList(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief Helper to compute centroid
     */
    Eigen::Vector3f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief Helper to compute max distance from centroid
     */
    float computeMaxDistance(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const Eigen::Vector3f& centroid);

    // Instance variables
    float last_error_;
    float last_registration_time_;
    float last_normalization_scale_;
};

} // namespace go_icp

#endif // GO_ICP_WRAPPER_HPP