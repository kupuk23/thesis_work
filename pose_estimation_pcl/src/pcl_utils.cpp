#include "pose_estimation_pcl/pcl_utils.hpp"
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <random>
#include <string>
#include <chrono>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/visualization/pcl_visualizer.h>



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


// Function to cluster a point cloud and color each cluster
ClusteringResult cluster_point_cloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size
) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create result struct
ClusteringResult result;
std::vector<pcl::PointIndices> cluster_indices;

// Create KdTree for searching
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
RCLCPP_INFO(logger, "Starting clustering with tolerance=%.3f, min_size=%d, max_size=%d",
        cluster_tolerance, min_cluster_size, max_cluster_size);

ec.extract(cluster_indices);

// Check if we found any clusters
if (cluster_indices.empty()) {
    RCLCPP_WARN(logger, "No clusters found in the point cloud");
    result.colored_cloud = input_cloud;  // Return original cloud if no clusters found
    return result;
}
RCLCPP_INFO(logger, "Found %ld clusters", cluster_indices.size());

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
    
    RCLCPP_INFO(logger, "Cluster %ld: %ld points, color RGB(%d,%d,%d)",
            i, result.individual_clusters[i]->size(), color[0], color[1], color[2]);
    
    // Add this cluster to the output cloud
    *result.colored_cloud += *result.individual_clusters[i];
}

// Measure and report execution time
auto end_time = std::chrono::high_resolution_clock::now();
double execution_time = std::chrono::duration<double>(end_time - start_time).count();

RCLCPP_INFO(logger, "Clustering completed in %.3f seconds", execution_time);
RCLCPP_INFO(logger, "Output cloud has %ld points across %ld clusters", 
        result.colored_cloud->size(), cluster_indices.size());

return result;
}


PlaneSegmentationResult detect_and_remove_planes(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
    const rclcpp::Logger& logger,
    bool colorize_planes, size_t min_plane_points,float min_remaining_percent ,int max_planes) {
    
    PlaneSegmentationResult result;
    result.remaining_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    result.planes_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    result.largest_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Initialize working cloud as a copy of input
    size_t largest_plane_size = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*input_cloud));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr with_largest_plane_removed(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    // Plane segmentation setup
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);  // 2cm threshold
    seg.setMaxIterations(100);
    
    int plane_count = 0;
    size_t remaining_points = working_cloud->size();
    
    

    
    RCLCPP_INFO(logger, "Starting plane detection on cloud with %zu points", remaining_points);
    
    // Detect all planes but only update the working cloud at the end
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*working_cloud));
    
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
        
        // Extract the plane
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
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
        
        // If this is the largest plane so far, save it and the result of removing it
    if (inliers->indices.size() > largest_plane_size) {
        largest_plane_size = inliers->indices.size();
        *largest_plane = *plane_cloud;
        
        // Save the cloud with this plane removed
        extract.setNegative(true);
        extract.filter(*with_largest_plane_removed);
    }

        
        if (colorize_planes) {
            // Get color from our fixed set
            const std::array<uint8_t, 3>& color = DEFAULT_COLORS[(plane_count - 1) % DEFAULT_COLORS.size()];
            uint8_t r = color[0];
            uint8_t g = color[1];
            uint8_t b = color[2];
            
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
        
        // RCLCPP_INFO(logger, "Plane %d: extracted %lu points (%.1f%% of original)",
        //            plane_count, inliers->indices.size(),
        //            100.0f * inliers->indices.size() / input_cloud->size());
        
        // Print plane equation: ax + by + cz + d = 0
        // RCLCPP_INFO(logger, "Plane equation: %.2fx + %.2fy + %.2fz + %.2f = 0",
        //            coefficients->values[0], coefficients->values[1],
        //            coefficients->values[2], coefficients->values[3]);
        
        // Update the working cloud for next iteration
        working_cloud = remaining_cloud;
    }
    
    // Now that we've found all planes and identified the largest one,
    // remove only the largest plane from the input cloud
    if (largest_plane_size > 0) {
        *result.largest_plane_cloud = *largest_plane;
        *result.remaining_cloud = *with_largest_plane_removed;
        // RCLCPP_INFO(logger, "Removed largest plane with %lu points. Remaining cloud has %lu points.",
        //            largest_plane_size, result.remaining_cloud->size());
    } else {
        // No planes found, return the original cloud
        *result.remaining_cloud = *input_cloud;
        RCLCPP_INFO(logger, "No planes found to remove");
    }
    
    RCLCPP_INFO(logger, "Plane detection complete. Found %d planes, visualizing %lu points in planes cloud.",
               plane_count, result.planes_cloud->size());
    
    return result;
}

std::vector<ClusterFeatures> computeFPFHFeatures(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters,
    float normal_radius,
    float feature_radius,
    int num_threads,
    bool visualize_normals,
    const rclcpp::Logger& logger) {
    
    std::vector<ClusterFeatures> results;
    results.reserve(clusters.size());
    
    RCLCPP_INFO(logger, "Computing FPFH features for %ld clusters", clusters.size());
    
    // Process each cluster
    for (size_t i = 0; i < clusters.size(); ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert RGB to XYZ (FPFH works with XYZ points)
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        xyz_cloud->points.resize(clusters[i]->size());
        xyz_cloud->width = clusters[i]->width;
        xyz_cloud->height = clusters[i]->height;
        xyz_cloud->is_dense = clusters[i]->is_dense;
        
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
        
        // Create a new result for this cluster
        ClusterFeatures cluster_result;
        
        try {
            // Create search method
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            
            // Estimate normals using OMP version for parallel processing
            pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
            if (num_threads > 0) normal_estimator.setNumberOfThreads(num_threads);
            
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            
            normal_estimator.setInputCloud(xyz_cloud);
            normal_estimator.setSearchMethod(tree);
            normal_estimator.setRadiusSearch(normal_radius);
            normal_estimator.compute(*normals);

            // Visualize normals if requested
            if (visualize_normals) {
                RCLCPP_INFO(logger, "Visualizing normals for cluster %ld", i);
                std::string window_name = "Normals for Cluster " + std::to_string(i);
                visualizeNormals(xyz_cloud, normals, 0.02, 3, window_name);
            }
            
            // Compute FPFH features using OMP version
            pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
            if (num_threads > 0) fpfh.setNumberOfThreads(num_threads);
            
            fpfh.setInputCloud(xyz_cloud);
            fpfh.setInputNormals(normals);
            fpfh.setSearchMethod(tree);
            fpfh.setRadiusSearch(feature_radius);
            fpfh.compute(*(cluster_result.fpfh_features));
            
            // Compute average FPFH descriptor
            for (int j = 0; j < 33; ++j) {
                cluster_result.average_fpfh.histogram[j] = 0;
            }
            
            for (size_t j = 0; j < cluster_result.fpfh_features->size(); ++j) {
                for (int k = 0; k < 33; ++k) {
                    cluster_result.average_fpfh.histogram[k] += 
                        cluster_result.fpfh_features->points[j].histogram[k];
                }
            }
            
            for (int j = 0; j < 33; ++j) {
                cluster_result.average_fpfh.histogram[j] /= cluster_result.fpfh_features->size();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double>(end_time - start_time).count();
            
            RCLCPP_INFO(logger, "Cluster %ld: Computed %ld FPFH features in %.3f seconds",
                      i, cluster_result.fpfh_features->size(), execution_time);
            
            results.push_back(cluster_result);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Error computing features for cluster %ld: %s", i, e.what());
        }
    }
    
    RCLCPP_INFO(logger, "FPFH computation complete for %ld of %ld clusters", 
               results.size(), clusters.size());
    
    return results;
}

} // namespace pcl_utils