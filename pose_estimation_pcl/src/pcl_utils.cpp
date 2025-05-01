#include "pose_estimation_pcl/pcl_utils.hpp"
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <random>
#include <string>
#include <chrono>

#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/features/features.hpp>


namespace pcl_utils {


void visualizeNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float normal_length,
    int point_size,
    const std::string& window_name,
    bool blocking) {
    
    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_name));
    viewer->setBackgroundColor(0, 0, 0);
    
    // Add point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color_handler, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "cloud");
    
    // Add normals
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, normal_length, "normals");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "normals");
    
    // Add coordinate system
    viewer->addCoordinateSystem(0.1);
    
    // Set camera position
    viewer->initCameraParameters();
    
    // Display the visualizer
    std::cout << "Press 'q' to close the visualizer window when done viewing." << std::endl;
    
    if (blocking) {
        viewer->spin();
    } else {
        // Just display and return immediately
        viewer->spinOnce(100);
    }
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertPointCloud2ToPCL(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Reserve space for the points
    cloud->width = cloud_msg->width;
    cloud->height = cloud_msg->height;
    cloud->is_dense = false;
    cloud->points.resize(cloud_msg->width * cloud_msg->height);
    
    // Create iterators for all fields
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(*cloud_msg, "rgb");
    
    // Fill in the PCL cloud data using iterators
    int point_idx = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
        pcl::PointXYZRGB& pcl_point = cloud->points[point_idx];
        pcl_point.x = *iter_x;
        pcl_point.y = *iter_y;
        pcl_point.z = *iter_z;
        
        // Get the float RGB value and reinterpret as int
        float rgb_float = *iter_rgb;
        int rgb_int = *reinterpret_cast<int*>(&rgb_float);
        
        // Extract RGB components
        uint8_t r = (rgb_int >> 16) & 0xFF;
        uint8_t g = (rgb_int >> 8) & 0xFF;
        uint8_t b = rgb_int & 0xFF;
        
        // Pack into PCL RGB format
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        pcl_point.rgb = *reinterpret_cast<float*>(&rgb);
        
        point_idx++;
    }
    
    return cloud;
}

void saveToPCD(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
    const std::string& filename,
    const rclcpp::Logger& logger) {
    
    try {
        RCLCPP_INFO(logger, "Saving cloud to %s", filename.c_str());
        pcl::io::savePCDFileBinary(filename, *cloud);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Failed to save PCD file: %s", e.what());
    }
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadModelPCD(
    const std::string& filename,
    const rclcpp::Logger& logger) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *model_cloud) == -1) {
        RCLCPP_ERROR(logger, "Failed to load model PCD file: %s", filename.c_str());
    } else {
        RCLCPP_INFO(logger, "Loaded model PCD with %ld points", model_cloud->size());
    }
    return model_cloud;
}


ClusteringResult cluster_point_cloud_gpu(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size,
    bool debug_time,
    const rclcpp::Logger& logger)
{
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create result struct
    ClusteringResult result;
    result.colored_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Convert XYZRGB to XYZ (GPU processing needs XYZ)
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    xyz_cloud->resize(input_cloud->size());
    
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        xyz_cloud->points[i].x = input_cloud->points[i].x;
        xyz_cloud->points[i].y = input_cloud->points[i].y;
        xyz_cloud->points[i].z = input_cloud->points[i].z;
    }
    
    // Upload point cloud to GPU
    pcl::gpu::Octree::PointCloud cloud_device;
    cloud_device.upload(xyz_cloud->points);
    
    // Create GPU octree using a shared pointer
    pcl::gpu::Octree::Ptr octree_device(new pcl::gpu::Octree);
    octree_device->setCloud(cloud_device);
    octree_device->build();
    
    // Run clustering on GPU
    pcl::gpu::EuclideanClusterExtraction<pcl::PointXYZ> gec;
    gec.setClusterTolerance(cluster_tolerance);
    gec.setMinClusterSize(min_cluster_size);
    gec.setMaxClusterSize(max_cluster_size);
    gec.setSearchMethod(octree_device);
    gec.setHostCloud(xyz_cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    gec.extract(cluster_indices);
    
    // Check if we found any clusters
    if (cluster_indices.empty()) {
        RCLCPP_WARN(logger, "No clusters found in the point cloud");
        result.colored_cloud = input_cloud;  // Return original cloud if no clusters found
        return result;
    }
    
    // Reserve space for individual clusters
    result.individual_clusters.resize(cluster_indices.size());
    
    // Extract and color each cluster
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        // Get a color for this cluster
        const std::array<uint8_t, 3>& color = DEFAULT_COLORS[i % DEFAULT_COLORS.size()];
        
        // Create a cloud for this cluster
        result.individual_clusters[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // Extract points for this cluster and color them
        for (const auto& idx : cluster_indices[i].indices) {
            pcl::PointXYZRGB colored_point = input_cloud->points[idx];
            
            // Set the RGB color
            colored_point.r = color[0];
            colored_point.g = color[1];
            colored_point.b = color[2];
            
            result.individual_clusters[i]->push_back(colored_point);
        }
        
        result.individual_clusters[i]->width = result.individual_clusters[i]->size();
        result.individual_clusters[i]->height = 1;
        result.individual_clusters[i]->is_dense = true;
        
        // Add this cluster to the output cloud
        *result.colored_cloud += *result.individual_clusters[i];
    }
    
    // Measure and report execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (debug_time) RCLCPP_INFO(logger, "GPU clustering completed in %.3f seconds", execution_time);
    
    return result;
}

// Function to cluster a point cloud and color each cluster
ClusteringResult cluster_point_cloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size,
    bool debug_time,
    const rclcpp::Logger& logger
) {

    // return cluster_point_cloud_gpu(input_cloud, cluster_tolerance, 
    //     min_cluster_size, max_cluster_size, logger);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create result struct
    ClusteringResult result;
    std::vector<pcl::PointIndices> cluster_indices;

    // Create KdTree for searching
    // float resolution = 0.01f;  // 1cm resolution for the octree
    // pcl::search::Octree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::Octree<pcl::PointXYZRGB>(resolution));
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(input_cloud);

    // Setup clustering
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input_cloud);

    // Perform clustering
    // RCLCPP_INFO(logger, "Starting clustering with tolerance=%.3f, min_size=%d, max_size=%d",
    //         cluster_tolerance, min_cluster_size, max_cluster_size);

    ec.extract(cluster_indices);

    // Check if we found any clusters
    if (cluster_indices.empty()) {
        RCLCPP_WARN(logger, "No clusters found in the point cloud");
        result.colored_cloud = input_cloud;  // Return original cloud if no clusters found
        return result;
    }
    // RCLCPP_INFO(logger, "Found %ld clusters", cluster_indices.size());

    // Reserve space for individual clusters
    result.individual_clusters.resize(cluster_indices.size());

    // Extract and color each cluster
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        // Get a color for this cluster
        const std::array<uint8_t,3>& color = DEFAULT_COLORS[i % DEFAULT_COLORS.size()];
        
        // Create a cloud for this cluster
        result.individual_clusters[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // Extract points for this cluster and color them
        for (const auto& idx : cluster_indices[i].indices) {
            pcl::PointXYZRGB colored_point = input_cloud->points[idx];
            
            // Set the RGB color
            colored_point.r = color[0];
            colored_point.g = color[1];
            colored_point.b = color[2];
            
            result.individual_clusters[i]->push_back(colored_point);
        }
        
        result.individual_clusters[i]->width = result.individual_clusters[i]->size();
        result.individual_clusters[i]->height = 1;
        result.individual_clusters[i]->is_dense = true;
        
        // RCLCPP_INFO(logger, "Cluster %ld: %ld points, color RGB(%d,%d,%d)",
        //         i, result.individual_clusters[i]->size(), color[0], color[1], color[2]);
        
        // Add this cluster to the output cloud
        *result.colored_cloud += *result.individual_clusters[i];
    }

    // Measure and report execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    if (debug_time) RCLCPP_INFO(logger, "Clustering completed in %.3f seconds", execution_time);
    // RCLCPP_INFO(logger, "Output cloud has %ld points across %ld clusters", 
    //         result.colored_cloud->size(), cluster_indices.size());

    return result;
    }


    bool check_floor_plane(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& plane_cloud,
        const pcl::ModelCoefficients::Ptr& coefficients,
        const Eigen::Matrix4f& camera_to_map_transform,
        const rclcpp::Logger& logger) {
        
        if (!plane_cloud || plane_cloud->empty() || coefficients->values.size() < 4) {
            RCLCPP_WARN(logger, "Invalid plane data for floor check");
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
            Eigen::Matrix3f rotation_matrix = camera_to_map_transform.block<3, 3>(0, 0);
            
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
                Eigen::Vector4f centroid_map = camera_to_map_transform * centroid_camera;
                
                // Extract camera position in map frame (translation part of the transform)
                Eigen::Vector3f camera_position = camera_to_map_transform.block<3, 1>(0, 3);
                
                // If the plane is below the camera/robot, it's likely the floor
                bool below_camera = centroid_map[2] < camera_position[2];
                
                if (below_camera) {
                    RCLCPP_INFO(logger, "Detected floor plane! Normal angle to Z: %.2f degrees", angle_degrees);
                    return true;
                } else {
                    RCLCPP_INFO(logger, "Detected ceiling plane! Normal angle to Z: %.2f degrees", angle_degrees);
                    return false;
                }
            } else {
                RCLCPP_DEBUG(logger, "Detected vertical plane. Normal angle to Z: %.2f degrees", angle_degrees);
                return false;
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Error in floor plane check: %s", e.what());
            return false;
        }
    }


PlaneSegmentationResult detect_and_remove_planes(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    bool colorize_planes, size_t min_plane_points,float min_remaining_percent ,
    int max_planes, float dist_threshold, int max_iterations, bool debug_time, 
    const Eigen::Matrix4f& camera_to_map_transform) {
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PlaneSegmentationResult result;
    result.remaining_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    result.planes_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    result.largest_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Initialize working cloud as a copy of input
    size_t largest_plane_size = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr with_largest_plane_removed(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*input_cloud));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*working_cloud));
    

    // Plane segmentation setup
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr largest_plane_indices(new pcl::PointIndices());
    pcl::PointIndices::Ptr floor_plane_indices(new pcl::PointIndices());
    bool is_floor = false;

    // Extract the plane
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(dist_threshold);  // 2cm threshold
    seg.setMaxIterations(max_iterations);
    
    int plane_count = 0;
    size_t remaining_points = working_cloud->size();
    
    // RCLCPP_INFO(logger, "Starting plane detection on cloud with %zu points", remaining_points);
    
    // Detect all planes but only update the working cloud at the end
    
    while (plane_count < max_planes && 
        remaining_points > (min_remaining_percent * input_cloud->size()) &&
        remaining_points > min_plane_points) {
 
        
        // Segment the next planar component
        seg.setInputCloud(working_cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < min_plane_points) {
            RCLCPP_INFO(logger, "No more significant planes found");
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
        is_floor = check_floor_plane(plane_cloud, coefficients, camera_to_map_transform, logger);
        // check if the segment is represent the floor
        if (is_floor) {
            // RCLCPP_INFO(logger, "Detected floor plane, skipping");
            *floor_plane_indices = *inliers;
        }
        
        // If this is the largest plane so far, save it and the result of removing it
        else if (inliers->indices.size() > largest_plane_size) {
            largest_plane_size = inliers->indices.size();
            *largest_plane = *plane_cloud;
            *largest_plane_indices = *inliers;
            
            // Save the cloud with this plane removed
            // extract.setNegative(true);
            // extract.filter(*with_largest_plane_removed);
        }

        
        if (colorize_planes) {
            // Get color from our fixed set
            const std::array<uint8_t, 3>& color = DEFAULT_COLORS[(plane_count - 1) % DEFAULT_COLORS.size()];
            uint8_t r = is_floor ? 255 : color[0];
            uint8_t g = is_floor ? 255 : color[1];
            uint8_t b = is_floor ? 0 : color[2];
            
            // RCLCPP_INFO(logger, "Coloring plane %d with RGB(%d,%d,%d)", plane_count, r, g, b);
            
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
            *result.planes_cloud += *colored_plane;
        }

        // Print plane equation: ax + by + cz + d = 0
        // RCLCPP_INFO(logger, "Plane equation: %.2fx + %.2fy + %.2fz + %.2f = 0",
        //            coefficients->values[0], coefficients->values[1],
        //            coefficients->values[2], coefficients->values[3]);
        
        // Update the working cloud for next iteration
        working_cloud = remaining_cloud;


    }
    
    // we found all planes and identified the largest one,
    // remove only the largest plane from the input cloud
    if (largest_plane_size > 0) {
        

        *result.largest_plane_cloud = *largest_plane;
        extract.setInputCloud(input_cloud);
        extract.setIndices(largest_plane_indices);
        extract.setNegative(true);
        extract.filter(*result.remaining_cloud);

        // if (floor_plane_indices->indices.size() > 0) {
        //     // Remove the floor plane from the remaining cloud
        //     extract.setInputCloud(result.remaining_cloud);
        //     extract.setIndices(floor_plane_indices);
        //     extract.setNegative(true);
        //     extract.filter(*result.remaining_cloud);
        // }
        // RCLCPP_INFO(logger, "Removed largest plane with %lu points. Remaining cloud has %lu points.",
        //            largest_plane_size, result.remaining_cloud->size());
    } else {
        // No planes found, return the original cloud
        *result.remaining_cloud = *input_cloud;
        RCLCPP_INFO(logger, "No planes found to remove");
    }
    
    // RCLCPP_INFO(logger, "Plane detection complete. Found %d planes, visualizing %lu points in planes cloud.",
    //            plane_count, result.planes_cloud->size());
    
    // Calculate and log execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    if (debug_time) RCLCPP_INFO(logger, "Plane detection executed in %.3f seconds", execution_time);
    
    return result;
}


std::vector<ClusterFeatures> computeFPFHFeaturesGPU(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters,
    float normal_radius,
    float feature_radius,
    bool visualize_normals,
    const rclcpp::Logger& logger) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<ClusterFeatures> results;
    results.reserve(clusters.size());
    
    RCLCPP_INFO(logger, "Computing FPFH features using GPU for %ld clusters", clusters.size());
    
    // Process each cluster
    for (size_t i = 0; i < clusters.size(); ++i) {
        auto cluster_start_time = std::chrono::high_resolution_clock::now();
        
        // Skip empty clusters
        if (clusters[i]->empty()) {
            RCLCPP_WARN(logger, "Cluster %ld is empty, skipping", i);
            continue;
        }
        
        // Convert RGB to XYZ for GPU processing
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        xyz_cloud->resize(clusters[i]->size());
        
        for (size_t j = 0; j < clusters[i]->size(); ++j) {
            xyz_cloud->points[j].x = clusters[i]->points[j].x;
            xyz_cloud->points[j].y = clusters[i]->points[j].y;
            xyz_cloud->points[j].z = clusters[i]->points[j].z;
        }
        
        // Skip if cluster has too few points
        if (xyz_cloud->size() < 10) {
            RCLCPP_WARN(logger, "Cluster %ld has only %ld points, skipping feature computation", 
                       i, xyz_cloud->size());
            continue;
        }
        
        ClusterFeatures cluster_result;
        
        try {
            // Upload point cloud to GPU
            pcl::gpu::DeviceArray<pcl::PointXYZ> device_cloud;
            device_cloud.upload(xyz_cloud->points);
            
            // Create a GPU-based KdTree for search
            pcl::gpu::DeviceArray<pcl::PointXYZ> device_normals;
            pcl::gpu::NormalEstimation ne;
            ne.setInputCloud(device_cloud);
            ne.setRadiusSearch(normal_radius, INT_MAX); // Add max elements parameter
            ne.compute(device_normals);
            
            // Download normals for visualization (if needed)
            pcl::PointCloud<pcl::Normal>::Ptr host_normals;
            
            // if (visualize_normals) {
            //     host_normals.reset(new pcl::PointCloud<pcl::Normal>());
            //     host_normals->resize(device_normals.size());
            //     device_normals.download(host_normals->points);
                
            //     std::string window_name = "Normals for Cluster " + std::to_string(i);
            //     visualizeNormals(xyz_cloud, host_normals, 0.02, 3, window_name);
            // }
            
            // Compute FPFH features on GPU
            pcl::gpu::FPFHEstimation fpfh_gpu;
            pcl::gpu::DeviceArray2D<pcl::FPFHSignature33> device_features;
            fpfh_gpu.setInputCloud(device_cloud);
            fpfh_gpu.setInputNormals(device_normals);
            fpfh_gpu.setRadiusSearch(feature_radius, INT_MAX); // Add max elements parameter
            fpfh_gpu.compute(device_features);
            
            // Download features to host
            cluster_result.fpfh_features.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
            int rows = device_features.rows();
            int cols = device_features.cols();
            cluster_result.fpfh_features->resize(rows);
            
            // Download features from GPU to host using a proper host Array2D
            std::vector<pcl::FPFHSignature33> host_features;
            host_features.resize(rows);
            device_features.download(&host_features[0], cols * sizeof(pcl::FPFHSignature33));
            
            // Copy downloaded features to the point cloud
            for (int j = 0; j < rows; j++) {
                cluster_result.fpfh_features->points[j] = host_features[j];
            }
            
            // Initialize average descriptor to zeros
            std::fill_n(cluster_result.average_fpfh.histogram, 33, 0.0f);
            
            // Compute average and normalize individual features
            const size_t num_features = cluster_result.fpfh_features->size();
            
            // Parallel reduction for computing average (using CPU since data is now downloaded)
            #pragma omp parallel
            {
                pcl::FPFHSignature33 local_avg;
                std::fill_n(local_avg.histogram, 33, 0.0f);
                
                #pragma omp for
                for (size_t j = 0; j < num_features; ++j) {
                    // Get the current feature
                    pcl::FPFHSignature33& feature = cluster_result.fpfh_features->points[j];
                    
                    // Normalize the individual feature
                    float point_sum = 0.0f;
                    for (int k = 0; k < 33; ++k) {
                        point_sum += feature.histogram[k];
                    }
                    
                    if (point_sum > 0) {
                        float inv_sum = 1.0f / point_sum;
                        for (int k = 0; k < 33; ++k) {
                            feature.histogram[k] *= inv_sum;  // Normalize
                            local_avg.histogram[k] += feature.histogram[k];
                        }
                    }
                }
                
                #pragma omp critical
                {
                    for (int k = 0; k < 33; ++k) {
                        cluster_result.average_fpfh.histogram[k] += local_avg.histogram[k];
                    }
                }
            }
            
            // Finalize average by dividing by number of features
            if (num_features > 0) {
                float inv_size = 1.0f / static_cast<float>(num_features);
                for (int j = 0; j < 33; ++j) {
                    cluster_result.average_fpfh.histogram[j] *= inv_size;
                }
            }
            
            auto cluster_end_time = std::chrono::high_resolution_clock::now();
            double cluster_time = std::chrono::duration<double>(cluster_end_time - cluster_start_time).count();
            
            RCLCPP_INFO(logger, "Cluster %ld: GPU computed %ld FPFH features in %.3f seconds",
                       i, num_features, cluster_time);
            
            results.push_back(cluster_result);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Error in GPU feature computation for cluster %ld: %s", i, e.what());
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    RCLCPP_INFO(logger, "GPU FPFH computation executed in %.3f seconds", execution_time);
    
    return results;
}

std::vector<ClusterFeatures> computeFPFHFeatures(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters,
    float normal_radius,
    float feature_radius,
    int num_threads,
    bool visualize_normals,
    bool debug_time,
    const rclcpp::Logger& logger) {

    // return computeFPFHFeaturesGPU(clusters, normal_radius, feature_radius, 
    //         visualize_normals, logger);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<ClusterFeatures> results;
    results.reserve(clusters.size());
    
    // RCLCPP_INFO(logger, "Computing FPFH features for %ld clusters", clusters.size());
    
    // Pre-create reusable objects
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    if (num_threads > 0) normal_estimator.setNumberOfThreads(num_threads);
    normal_estimator.setRadiusSearch(normal_radius);
    
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    if (num_threads > 0) fpfh.setNumberOfThreads(num_threads);
    fpfh.setRadiusSearch(feature_radius);
    
    // Process each cluster
    for (size_t i = 0; i < clusters.size(); ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Skip empty clusters
        if (clusters[i]->empty()) {
            RCLCPP_WARN(logger, "Cluster %ld is empty, skipping", i);
            continue;
        }
        
        // Convert RGB to XYZ efficiently
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        xyz_cloud->resize(clusters[i]->size());
        xyz_cloud->width = clusters[i]->width;
        xyz_cloud->height = clusters[i]->height;
        xyz_cloud->is_dense = clusters[i]->is_dense;
        
        // Use memcpy for faster copy of XYZ data
        for (size_t j = 0; j < clusters[i]->size(); ++j) {
            // Direct assignment is faster than creating a temporary point
            xyz_cloud->points[j].x = clusters[i]->points[j].x;
            xyz_cloud->points[j].y = clusters[i]->points[j].y;
            xyz_cloud->points[j].z = clusters[i]->points[j].z;
        }
        
        // Skip if cluster has too few points
        if (xyz_cloud->size() < 10) {
            RCLCPP_WARN(logger, "Cluster %ld has only %ld points, skipping feature computation", 
                        i, xyz_cloud->size());
            continue;
        }
        
        // Create a new result for this cluster
        ClusterFeatures cluster_result;
        
        try {
            // Reset the KdTree for this cluster
            tree->setInputCloud(xyz_cloud);
            
            // Compute normals
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            normal_estimator.setInputCloud(xyz_cloud);
            normal_estimator.setSearchMethod(tree);
            normal_estimator.compute(*normals);
            
            // Visualize normals if requested
            if (visualize_normals) {
                // RCLCPP_INFO(logger, "Visualizing normals for cluster %ld", i);
                std::string window_name = "Normals for Cluster " + std::to_string(i);
                visualizeNormals(xyz_cloud, normals, 0.02, 3, window_name);
            }
            
            // Compute FPFH features
            fpfh.setInputCloud(xyz_cloud);
            fpfh.setInputNormals(normals);
            fpfh.setSearchMethod(tree);
            fpfh.compute(*(cluster_result.fpfh_features));
            
            // Early exit if no features were computed
            if (cluster_result.fpfh_features->empty()) {
                RCLCPP_WARN(logger, "No FPFH features computed for cluster %ld", i);
                continue;
            }
            
            // Initialize average descriptor to zeros
            std::fill_n(cluster_result.average_fpfh.histogram, 33, 0.0f);
            
            // Compute average and normalize individual features in one pass
            const size_t num_features = cluster_result.fpfh_features->size();
            for (size_t j = 0; j < num_features; ++j) {
                // Get the current feature
                pcl::FPFHSignature33& feature = cluster_result.fpfh_features->points[j];
                
                // Normalize the individual feature
                float point_sum = 0.0f;
                for (int k = 0; k < 33; ++k) {
                    point_sum += feature.histogram[k];
                }
                
                if (point_sum > 0) {
                    // float inv_sum = 1.0f / point_sum;
                    for (int k = 0; k < 33; ++k) {
                        feature.histogram[k] /= point_sum;  // Normalize
                        cluster_result.average_fpfh.histogram[k] += feature.histogram[k];
                    }
                }
            }
            
            // Finalize average by dividing by number of features
            if (num_features > 0) {
                float inv_size = 1.0f / static_cast<float>(num_features);
                for (int j = 0; j < 33; ++j) {
                    cluster_result.average_fpfh.histogram[j] *= inv_size;
                }
            }
            
            // Already normalized since we normalized individual features first
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double>(end_time - start_time).count();
            
            // RCLCPP_INFO(logger, "Cluster %ld: Computed %ld normalized FPFH features in %.3f seconds",
            //           i, num_features, execution_time);
            
            results.push_back(cluster_result);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Error computing features for cluster %ld: %s", i, e.what());
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    if (debug_time) RCLCPP_INFO(logger, "FPFH computation executed in %.3f seconds", execution_time);
    
    // RCLCPP_INFO(logger, "FPFH computation complete for %ld of %ld clusters", 
    //            results.size(), clusters.size());
    
    return results;
}

HistogramMatchingResult findBestClusterByHistogram(
    const ClusterFeatures& model_features,
    const std::vector<ClusterFeatures>& cluster_features,
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cluster_clouds,
    float similarity_threshold,
    bool debug_time,
    const rclcpp::Logger& logger)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    HistogramMatchingResult result;
    result.cluster_similarities.resize(cluster_features.size(), 0.0f);
    
    // RCLCPP_INFO(logger, "Matching model with %ld clusters using histogram matching", cluster_features.size());
    
    // Track best match information locally instead of in the result struct
    int best_cluster_index = -1;
    float best_similarity_score = 0.0f;
    
    // Process each cluster
    for (size_t i = 0; i < cluster_features.size(); ++i) {
        // Skip if the cluster has no features
        if (cluster_features[i].fpfh_features->empty()) {
            RCLCPP_WARN(logger, "Cluster %ld has no FPFH features, skipping", i);
            continue;
        }
        
        // Compare average FPFH histograms using histogram intersection
        float similarity = 0.0f;
        for (int j = 0; j < 33; ++j) {
            similarity += std::min(
                model_features.average_fpfh.histogram[j],
                cluster_features[i].average_fpfh.histogram[j]
            );
        }
        result.cluster_similarities[i] = similarity;
        
        // Update best match if this cluster has better similarity
        if (similarity > best_similarity_score) {
            best_cluster_index = i;
            best_similarity_score = similarity;
        }
    }
    
    // Check if the best match exceeds the threshold
    if (best_similarity_score >= similarity_threshold && 
        best_cluster_index >= 0 && 
        best_cluster_index < static_cast<int>(cluster_clouds.size())) {
        
        result.best_matching_cluster = cluster_clouds[best_cluster_index];
        // RCLCPP_INFO(logger, "Best matching cluster: %d with similarity score: %.4f", 
        //            best_cluster_index, best_similarity_score);
    } else {
        RCLCPP_WARN(logger, "Best cluster (idx: %d) similarity %.4f below threshold %.4f, rejecting match",
                  best_cluster_index, best_similarity_score, similarity_threshold);
        // No valid match found - keep result.best_matching_cluster as nullptr
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    if (debug_time) RCLCPP_INFO(logger, "Finding best cluster executed in %.3f seconds", execution_time);
    
    return result;
}

pcl_utils::ClusterFeatures loadAndComputeModelFeatures(
    const std::string& object_name,
    float normal_radius,
    float feature_radius,
    int num_threads,
    bool visualize_normals,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& model_cloud_out,
    const rclcpp::Logger& logger) {
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Determine file path based on object name
    std::string model_path;
    if (object_name == "grapple") {
        model_path = "/home/tafarrel/o3d_logs/grapple_fixture_v2.pcd";
    } else if (object_name == "handrail") {
        model_path = "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd";
    } else if (object_name == "docking_st") {
        model_path = "/home/tafarrel/o3d_logs/astrobee_dock_ds.pcd";
    } else {
        RCLCPP_ERROR(logger, "Unknown object name: %s", object_name.c_str());
        return ClusterFeatures();  // Return empty features
    }
    
    // Load model point cloud
    model_cloud_out = loadModelPCD(model_path, logger);
    if (model_cloud_out->empty()) {
        RCLCPP_ERROR(logger, "Failed to load model for %s", object_name.c_str());
        return ClusterFeatures();  // Return empty features
    }
    
    // Create vector with single model
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> model_vector = {model_cloud_out};
    
    // Compute features
    std::vector<ClusterFeatures> features = computeFPFHFeatures(
        model_vector,
        normal_radius,
        feature_radius,
        num_threads,
        visualize_normals,
        false,  // debug_time
        logger
    );
    
    // Calculate and log execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    // RCLCPP_INFO(logger, "Model feature computation for %s executed in %.3f seconds", 
    //            object_name.c_str(), execution_time);
    
    // Return the first (and only) feature set or throw an exception if none computed
if (features.empty()) {
    std::string error_msg = "CRITICAL ERROR: Failed to compute features for model " + object_name;
    RCLCPP_ERROR(logger, "%s", error_msg.c_str());
    throw std::runtime_error(error_msg);  // Throw exception instead of returning empty features
}
    
    return features[0];
}

} // namespace pcl_utils