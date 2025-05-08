#include "pose_estimation_pcl/plane_segmentation.hpp"
#include "pose_estimation_pcl/pcl_utils.hpp"


#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <chrono>
#include <cmath>
#include <limits>

namespace pose_estimation {

PlaneSegmentation::PlaneSegmentation(
    rclcpp::Logger logger,
    const Config& config)
    : logger_(logger), config_(config)
{
    RCLCPP_INFO(logger_, "Plane segmentation initialized");
}

void PlaneSegmentation::setTransform(const Eigen::Matrix4f& transform) {
    camera_to_map_transform_ = transform;
}

void PlaneSegmentation::setConfig(const Config& config) {
    config_ = config;
}

const PlaneSegmentation::Config& PlaneSegmentation::getConfig() const {
    return config_;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PlaneSegmentation::removeMainPlanes(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud) {

    // only run when the camera_to_map_transform_ is not zeros
    if (camera_to_map_transform_.isZero()) {
        RCLCPP_WARN(logger_, "Camera to map transform is not set, skipping plane segmentation");
        return input_cloud;
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize result structure
    last_result_.remaining_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    last_result_.planes_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    last_result_.largest_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Initialize working cloud as a copy of input
    size_t largest_plane_size = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr with_largest_plane_removed(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*input_cloud));

    // Plane segmentation setup
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::ModelCoefficients::Ptr floor_coeff(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    float height_threshold = 0.0f;
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    bool is_floor = false;

    // Configure plane segmentation
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(config_.distance_threshold);
    seg.setMaxIterations(config_.max_iterations);
    
    int plane_count = 0;
    size_t remaining_points = working_cloud->size();
    
    // Detect planes
    while (plane_count < config_.max_planes && 
           remaining_points > (config_.min_remaining_percent * input_cloud->size()) && 
           remaining_points > config_.min_plane_points) {
        
        // Segment the next planar component
        seg.setInputCloud(working_cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < config_.min_plane_points) {
            RCLCPP_DEBUG(logger_, "No more significant planes found");
            break;
        }
        
        extract.setInputCloud(working_cloud);
        extract.setIndices(inliers);
        
        // Get the points in the plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(false);
        extract.filter(*plane_cloud);
        
        // Get remaining points (for next iteration)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
        
        plane_count++;
        
        if (config_.colorize_planes) {
            // Get color from fixed set
            const std::array<uint8_t, 3>& color = pcl_utils::DEFAULT_COLORS[(plane_count - 1) % pcl_utils::DEFAULT_COLORS.size()];
            uint8_t r = color[0];
            uint8_t g = color[1];
            uint8_t b = color[2];
            
            // Create a copy of the plane cloud with the selected color
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
            colored_plane->points.resize(plane_cloud->points.size());
            colored_plane->width = plane_cloud->width;
            colored_plane->height = plane_cloud->height;
            colored_plane->is_dense = plane_cloud->is_dense;
            
            // Explicitly set each point's RGB value
            for (size_t i = 0; i < plane_cloud->points.size(); ++i) {
                // Copy the xyz coordinates
                colored_plane->points[i].x = plane_cloud->points[i].x;
                colored_plane->points[i].y = plane_cloud->points[i].y;
                colored_plane->points[i].z = plane_cloud->points[i].z;
                
                // Set the RGB color
                colored_plane->points[i].r = r;
                colored_plane->points[i].g = g;
                colored_plane->points[i].b = b;
            }
            
            // Add this colored plane to our composite cloud
            *last_result_.planes_cloud += *colored_plane;
        }
        
        // Check if this is a floor plane
        is_floor = checkFloorPlane(plane_cloud, coefficients, height_threshold);
        
        // If floor plane, save coefficients
        if (is_floor) {
            floor_coeff->values = coefficients->values;
        }
        // If this is the largest non-floor plane, save it
        else if (inliers->indices.size() > largest_plane_size) {
            largest_plane_size = inliers->indices.size();
            *largest_plane = *plane_cloud;
            extract.setNegative(true);
            extract.filter(*with_largest_plane_removed);
        }
        
        // Update the working cloud for next iteration
        working_cloud = remaining_cloud;
        remaining_points = working_cloud->size();
    }
    
    // We found all planes and identified the largest one,
    // remove only the largest plane from the input cloud
    if (largest_plane_size > 0) {
        *last_result_.largest_plane_cloud = *largest_plane;
        *last_result_.remaining_cloud = *with_largest_plane_removed;
        
        // If we found a floor plane, filter points close to the floor
        if (floor_coeff->values.size() == 4) {
            filterFloorClouds(last_result_.remaining_cloud, floor_coeff, height_threshold);
        }
    } else {
        // No planes found, return the original cloud
        *last_result_.remaining_cloud = *input_cloud;
        RCLCPP_INFO(logger_, "No planes found to remove");
    }
    
    // Calculate and log execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (debug_time_) {
        RCLCPP_INFO(logger_, "Plane detection executed in %.3f seconds", execution_time);
    }
    
    return last_result_.remaining_cloud;
}

const PlaneSegmentationResult& PlaneSegmentation::getLastResult() const {
    return last_result_;
}

bool PlaneSegmentation::checkFloorPlane(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& plane_cloud,
    const pcl::ModelCoefficients::Ptr& coefficients,
    float& height_threshold) {
    
    if (!plane_cloud || plane_cloud->empty() || coefficients->values.size() < 4) {
        RCLCPP_WARN(logger_, "Invalid plane data for floor check");
        return false;
    }
    
    try {
        // Get the plane normal in camera frame (a, b, c from plane equation)
        Eigen::Vector3f normal_camera(
            coefficients->values[0],
            coefficients->values[1], 
            coefficients->values[2]
        );
        
        // Normalize the vector
        normal_camera.normalize();
        
        // Extract rotation matrix from the transform (upper-left 3x3)
        Eigen::Matrix3f rotation_matrix = camera_to_map_transform_.block<3, 3>(0, 0);
        
        // Transform normal from camera frame to map frame
        Eigen::Vector3f normal_map = rotation_matrix * normal_camera;
        
        // Z axis in map frame
        const Eigen::Vector3f z_axis(0.0, 0.0, 1.0);
        
        // Compute the dot product between the normal and z axis
        float dot_product = normal_map.dot(z_axis);
        
        // Compute the angle in degrees
        float angle_degrees = std::acos(std::abs(dot_product)) * 180.0f / M_PI;
        
        // Check if the plane is roughly horizontal (aligned with Z or negative Z)
        const float threshold_degrees = 15.0f;  // Planes within 15 degrees of horizontal
        bool is_horizontal = angle_degrees < threshold_degrees;
        
        if (is_horizontal) {
            // Check if it's below the camera (floor) or above (ceiling)
            // Use the centroid of the plane to determine this
            Eigen::Vector4f centroid_camera;
            pcl::compute3DCentroid(*plane_cloud, centroid_camera);
            
            // Transform centroid to map frame
            Eigen::Vector4f centroid_map = camera_to_map_transform_ * centroid_camera;
            
            // Extract camera position in map frame (translation part of the transform)
            Eigen::Vector3f camera_position = camera_to_map_transform_.block<3, 1>(0, 3);
            
            // If the plane is below the camera/robot, it's likely the floor
            bool below_camera = centroid_map[2] < camera_position[2];
            height_threshold = centroid_map[2] + 0.05f; // floor height + offset
            
            if (below_camera) {
                RCLCPP_DEBUG(logger_, "Detected floor plane! Normal angle to Z: %.2f degrees", angle_degrees);
                return true;
            } else {
                RCLCPP_DEBUG(logger_, "Detected ceiling plane! Normal angle to Z: %.2f degrees", angle_degrees);
                return false;
            }
        } else {
            RCLCPP_DEBUG(logger_, "Detected vertical plane. Normal angle to Z: %.2f degrees", angle_degrees);
            return false;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Error in floor plane check: %s", e.what());
        return false;
    }
}

void PlaneSegmentation::filterFloorClouds(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const pcl::ModelCoefficients::Ptr& coefficients,
    const float height_threshold) {

    // Transform the entire point cloud to map frame in one operation
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_map_frame(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud, *cloud_in_map_frame, camera_to_map_transform_);
    
    // Use PCL's PassThrough filter for efficient filtering on z axis
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud_in_map_frame);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(height_threshold, std::numeric_limits<float>::max());
    
    // Filter and store result
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_in_map(new pcl::PointCloud<pcl::PointXYZRGB>());
    pass.filter(*filtered_in_map);
    
    // Transform back to camera frame
    pcl::transformPointCloud(*filtered_in_map, *cloud, camera_to_map_transform_.inverse());
}

} // namespace pose_estimation