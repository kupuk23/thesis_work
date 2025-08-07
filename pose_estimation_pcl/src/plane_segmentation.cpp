#include "pose_estimation_pcl/plane_segmentation.hpp"
#include "pose_estimation_pcl/utils/pcl_utils.hpp"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <chrono>
#include <cmath>
#include <limits>

namespace pose_estimation
{

    PlaneSegmentation::PlaneSegmentation(
        const Config &config,
        rclcpp::Logger logger)
        : config_(config), logger_(logger)
    {
        RCLCPP_INFO(logger_, "Plane segmentation initialized");
    }

    void PlaneSegmentation::setTransform(const Eigen::Matrix4f &transform)
    {
        camera_to_map_transform_ = transform;
    }

    void PlaneSegmentation::setConfig(const Config &config)
    {
        config_ = config;
    }

    const PlaneSegmentation::Config &PlaneSegmentation::getConfig() const
    {
        return config_;
    }

    float PlaneSegmentation::measureFloorDist(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
                                              float eps_deg) // angular tolerance
    {
        auto temp = removeMainPlanes(input_cloud, Eigen::Vector3f(0, 0, 1), eps_deg, true);
        return floor_height_;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr PlaneSegmentation::removeMainPlanes(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
        const Eigen::Vector3f &axis, // e.g. (0,0,1) for floor
        float eps_deg,               // angular tolerance
        bool measuring_dist)
    {
        // only run when the camera_to_map_transform_ is not zeros
        if (camera_to_map_transform_.isZero())
        {
            RCLCPP_WARN(logger_, "Camera to map transform is not set, skipping plane segmentation");
            return input_cloud;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_map_frame(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cam(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::transformPointCloud(*input_cloud, *cloud_in_map_frame, camera_to_map_transform_);

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
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_in_map_frame));

        // Plane segmentation setup
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::ModelCoefficients::Ptr largest_coeff(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        float dist_threshold = 0.0f;
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        bool is_floor = false;
        float highest_floor = -std::numeric_limits<float>::max();

        // Configure plane segmentation
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(config_.distance_threshold);
        seg.setMaxIterations(config_.max_iterations);

        seg.setAxis(axis);                       // Set the axis to consider for plane detection
        seg.setEpsAngle(eps_deg * M_PI / 180.f); // Set angular tolerance

        // RCLCPP_INFO(logger_, "Plane segmentation started with axis: [%f, %f, %f] and eps_angle: %.2f degrees",
        //             axis.x(), axis.y(), axis.z(), eps_deg);

        int plane_count = 0;
        size_t remaining_points = working_cloud->size();

        // Detect planes
        while (plane_count < config_.max_planes &&
               remaining_points > (config_.min_remaining_percent * cloud_in_map_frame->size()) &&
               remaining_points > config_.min_plane_points)
        {

            try
            {
                // Segment the next planar component
                seg.setInputCloud(working_cloud);
                seg.segment(*inliers, *coefficients);
                extract.setInputCloud(working_cloud);
                extract.setIndices(inliers);
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(logger_, "Error during plane segmentation: %s", e.what());
                break;
            }

            if (inliers->indices.size() < config_.min_plane_points)
            {
                RCLCPP_DEBUG(logger_, "No more significant planes found");
                break;
            }
            else if (inliers->indices.empty())
            {
                RCLCPP_DEBUG(logger_, "No inliers found, breaking plane detection");
                break;
            }

            // Get the points in the plane
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            extract.setNegative(false);
            extract.filter(*plane_cloud);

            // Get remaining points (for next iteration)
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            extract.setNegative(true);
            extract.filter(*remaining_cloud);

            plane_count++;

            if (config_.colorize_planes)
            {
                // Get color from fixed set
                const std::array<uint8_t, 3> &color = pcl_utils::DEFAULT_COLORS[(plane_count - 1) % pcl_utils::DEFAULT_COLORS.size()];
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
                for (size_t i = 0; i < plane_cloud->points.size(); ++i)
                {
                    // Copy the xyz coordinates
                    colored_plane->points[i].x = plane_cloud->points[i].x;
                    colored_plane->points[i].y = plane_cloud->points[i].y;
                    colored_plane->points[i].z = plane_cloud->points[i].z;

                    // Set the RGB color
                    colored_plane->points[i].r = r;
                    colored_plane->points[i].g = g;
                    colored_plane->points[i].b = b;
                }
                
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_plane_cam(new pcl::PointCloud<pcl::PointXYZRGB>());
                // Add this colored plane to our composite cloud
                pcl::transformPointCloud(*colored_plane, *colored_plane_cam, camera_to_map_transform_.inverse());

                *last_result_.planes_cloud += *colored_plane_cam;
            }

            auto indices_size = inliers->indices.size();
            // RCLCPP_INFO(logger_, "Plane %d found with %zu points", plane_count, indices_size);

            // TODO :cek if condition ini, matiin full debug biar gampang ceknya. ini kebaca false terus
            if (inliers->indices.size() > largest_plane_size)
            {
                largest_plane_size = inliers->indices.size();
                *largest_plane = *plane_cloud;
                *largest_coeff = *coefficients;
                extract.setNegative(true);
                extract.filter(*with_largest_plane_removed);

                // if we are looking for a floor plane, append the floor coefficients and check for the highest floor, then get the height threshold
                if (getPlaneDistance(
                        plane_cloud, coefficients, dist_threshold, axis, 0.11f)) // 0.14f is the offset for floor height
                {
                    highest_floor = dist_threshold > highest_floor ? dist_threshold : highest_floor; // Store the highest floor
                    floor_height_ = highest_floor;
                    // RCLCPP_INFO(logger_, "floor plane found at height: %.2f", floor_height_);
                }
            }

            // Update the working cloud for next iteration
            working_cloud = remaining_cloud;
            remaining_points = working_cloud->size();
        }

        // We found all planes and identified the largest one,
        // remove only the largest plane from the input cloud
        if (largest_plane_size > 0)
        {

            // If we found a floor plane, filter points close to the floor
            if (axis == Eigen::Vector3f(0, 1, 0))
            {
                getPlaneDistance(
                    largest_plane, largest_coeff, dist_threshold, axis, 0.3f);
                filterCloudsByPlane(with_largest_plane_removed, "y", "up", dist_threshold);
            }
            else if (axis == Eigen::Vector3f(1, 0, 0))
            {
                getPlaneDistance(
                    largest_plane, largest_coeff, dist_threshold, axis, 0.3f);
                filterCloudsByPlane(with_largest_plane_removed, "x", "up", dist_threshold);
            }
            else if (axis == Eigen::Vector3f(0, 0, 1) && !measuring_dist)
            {
                filterCloudsByPlane(with_largest_plane_removed, "z", "up", highest_floor);
            }
            // filterCloudsByPlane(largest_plane, "x", "up", dist_threshold);

            // Transform the largest plane and remaining cloud back to camera frame
            pcl::transformPointCloud(*with_largest_plane_removed, *plane_filtered_cloud, camera_to_map_transform_.inverse());
            pcl::transformPointCloud(*largest_plane, *largest_plane_cam, camera_to_map_transform_.inverse());

            *last_result_.largest_plane_cloud = *largest_plane_cam;
            *last_result_.remaining_cloud = *plane_filtered_cloud;
        }
        else
        {
            // No planes found, return the original cloud
            *last_result_.remaining_cloud = *input_cloud;
            RCLCPP_INFO(logger_, "No planes found to remove");
            ;
        }

        // Calculate and log execution time
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end_time - start_time).count();

        if (debug_time_)
        {
            RCLCPP_INFO(logger_, "Plane detection executed in %.3f seconds", execution_time);
        }

        return last_result_.remaining_cloud;
    }

    const PlaneSegmentationResult &PlaneSegmentation::getLastResult() const
    {
        return last_result_;
    }

    bool PlaneSegmentation::getPlaneDistance(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud,
        const pcl::ModelCoefficients::Ptr &coefficients,
        float &dist_threshold,
        const Eigen::Vector3f &plane_axis,
        float offset)
    {
        if (!plane_cloud || plane_cloud->empty() || coefficients->values.size() < 4)
        {
            RCLCPP_WARN(logger_, "Invalid plane data for floor check");
            return false;
        }

        try
        {

            // Check if it's below the camera (floor) or above (ceiling)
            // Use the centroid of the plane to determine this
            Eigen::Vector4f centroid_plane_map;
            pcl::compute3DCentroid(*plane_cloud, centroid_plane_map);

            // Transform centroid to map frame
            // Eigen::Vector4f centroid_map = camera_to_map_transform_ * centroid_plane_map;
            Eigen::Vector3f camera_position = camera_to_map_transform_.block<3, 1>(0, 3);

            auto dist = centroid_plane_map[0];
            bool below_camera = false;
            if (plane_axis == Eigen::Vector3f(0, 0, 1))
            {
                dist = centroid_plane_map[2];
                below_camera = centroid_plane_map[2] < camera_position[2];
            }
            else if (plane_axis == Eigen::Vector3f(0, 1, 0))
                dist = centroid_plane_map[1];

            // if (offset < 0.0f)
            // {
            //     offset = -offset; // If the plane is above the camera, we need to subtract the offset
            // }
            offset = std::abs(offset);
            dist_threshold = dist + offset; // plane distance + offset

            return below_camera;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(logger_, "Error in floor plane check: %s", e.what());
            return false;
        }
    }

    void PlaneSegmentation::filterCloudsByPlane(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
        const std::string &plane_axis,
        const std::string &plane_direction,
        const float dist_threshold)
    {
        // Transform the entire point cloud to map frame in one operation
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_map_frame(new pcl::PointCloud<pcl::PointXYZRGB>());
        // pcl::transformPointCloud(*cloud, *cloud_in_map_frame, camera_to_map_transform_);

        // Use PCL's PassThrough filter for efficient filtering on specified axis
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName(plane_axis);

        RCLCPP_INFO(logger_, "Filtering points along %s axis with direction %s at threshold: %.2f",
                    plane_axis.c_str(), plane_direction.c_str(), dist_threshold);
        // Set filter limits based on direction
        if (plane_direction == "up")
        {
            pass.setFilterLimits(dist_threshold, std::numeric_limits<float>::max());
        }
        else
        {
            pass.setFilterLimits(-std::numeric_limits<float>::max(), dist_threshold);
        }

        // Filter and store result
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_in_map(new pcl::PointCloud<pcl::PointXYZRGB>());
        pass.filter(*filtered_in_map);

        cloud = filtered_in_map;

        // // Transform back to camera frame
        // pcl::transformPointCloud(*filtered_in_map, *cloud, camera_to_map_transform_.inverse());
    }

} // namespace pose_estimation