#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <thread>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>

// Include Go-ICP
#include "go_icp/jly_goicp.h"

// Function prototypes
void printUsage(const char* programName);
pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& filename);
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int maxPoints);
void normalizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& source_normalized,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& target_normalized,
    Eigen::Vector3f& source_centroid,
    Eigen::Vector3f& target_centroid,
    float& max_scale);
Eigen::Matrix4f denormalizeTransformation(
    const Eigen::Matrix4f& transform,
    const Eigen::Vector3f& source_centroid,
    const Eigen::Vector3f& target_centroid,
    float scale);
Eigen::Vector3f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
float computeMaxDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector3f& centroid);
std::vector<POINT3D> convertToPoint3DList(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
void visualizePointClouds(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const std::string& title = "Point Clouds");
void visualizeRegistrationResult(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const Eigen::Matrix4f& transform,
    const std::string& title = "Registration Result");

int main(int argc, char** argv) {
    bool visualize_before = false;
    bool visualize_after = false;
    int max_target_points = 3000;
    std::string source_filename = "/home/tafarrel/o3d_logs/grapple_fixture_down.pcd";
    std::string target_filename = "/home/tafarrel/o3d_logs/grapple_center.pcd";
    
    // Parse command line parameters
    
    
    if (argc >= 3) {
      source_filename = argv[1];
      target_filename = argv[2];
      
      for (int i = 3; i < argc; i++) {
          std::string arg = argv[i];
          if (arg == "--visualize-before" || arg == "-vb") {
              visualize_before = true;
          } else if (arg == "--visualize-after" || arg == "-va") {
              visualize_after = true;
          } else if (arg == "--max-points" || arg == "-mp") {
              if (i + 1 < argc) {
                  max_target_points = std::stoi(argv[++i]);
              }
          }
      }
    }

    
    std::cout << "Processing point cloud registration" << std::endl;
    std::cout << "Source file: " << source_filename << std::endl;
    std::cout << "Target file: " << target_filename << std::endl;
    
    // 1. Load point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = loadPointCloud(source_filename);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = loadPointCloud(target_filename);
    
    std::cout << "Source cloud size: " << source_cloud->size() << ", Target cloud size: " << target_cloud->size() << std::endl;
    
    // 2. Downsample target cloud if needed
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_downsampled = downsamplePointCloud(target_cloud, max_target_points);
    std::cout << "Target cloud after downsampling: " << target_downsampled->size() << " points" << std::endl;
    
    // 3. Visualize before registration if requested
    if (visualize_before) {
        std::cout << "Visualizing original point clouds..." << std::endl;
        visualizePointClouds(source_cloud, target_downsampled, "Original Point Clouds");
    }
    
    // 4. Normalize point clouds
    Eigen::Vector3f source_centroid, target_centroid;
    float max_scale;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_normalized(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_normalized(new pcl::PointCloud<pcl::PointXYZ>());
    
    normalizePointCloud(source_cloud, target_downsampled, source_normalized, target_normalized,
                        source_centroid, target_centroid, max_scale);
    
    std::cout << "Normalized point clouds with scale factor: " << max_scale << std::endl;
    
    if (visualize_before) {
        std::cout << "Visualizing normalized point clouds..." << std::endl;
        visualizePointClouds(source_normalized, target_normalized, "Normalized Point Clouds");
    }
    
    // 5. Convert to Go-ICP format
    std::vector<POINT3D> source_points = convertToPoint3DList(source_normalized);
    std::vector<POINT3D> target_points = convertToPoint3DList(target_normalized);
    
    // 6. Initialize Go-ICP
    GoICP goicp;
    ROTNODE rNode;
    TRANSNODE tNode;
    
    // Set rotation search space (full rotation space)
    rNode.a = -PI;
    rNode.b = -PI;
    rNode.c = -PI;
    rNode.w = 2*PI;
    
    // Set translation search space
    tNode.x = -1;
    tNode.y = -1;
    tNode.z = -1;
    tNode.w = 2.0;
    
    // Set Go-ICP parameters
    goicp.MSEThresh = 0.0005;  // Mean Square Error threshold
    goicp.trimFraction = 0.0;  // Trimming fraction (0.0 = no trimming)
    goicp.doTrim = (goicp.trimFraction >= 0.001);
    
    // Prepare data for Go-ICP
    goicp.pModel = source_points.data();
    goicp.Nm = source_points.size();
    goicp.pData = target_points.data();
    goicp.Nd = target_points.size();
    
    // Set DT parameters
    goicp.dt.SIZE = 50;
    goicp.dt.expandFactor = 2.0;
    
    // Set initial rotation and translation nodes
    goicp.initNodeRot = rNode;
    goicp.initNodeTrans = tNode;
    
    // 7. Run Go-ICP
    std::cout << "Building Distance Transform..." << std::endl;
    auto dt_start = std::chrono::high_resolution_clock::now();
    goicp.BuildDT();
    auto dt_end = std::chrono::high_resolution_clock::now();
    auto dt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dt_end - dt_start);
    std::cout << "Distance Transform built in " << dt_duration.count() / 1000.0 << " seconds" << std::endl;
    
    std::cout << "Starting registration..." << std::endl;
    auto reg_start = std::chrono::high_resolution_clock::now();
    float error = goicp.Register();
    auto reg_end = std::chrono::high_resolution_clock::now();
    auto reg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(reg_end - reg_start);
    
    std::cout << "Registration completed in " << reg_duration.count() / 1000.0 << " seconds with error: " << error << std::endl;
    
    // 8. Get optimal transformation
    Matrix optR = goicp.optR;
    Matrix optT = goicp.optT;
    
    // Create normalized transformation matrix
    Eigen::Matrix4f norm_transform = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            norm_transform(i, j) = optR.val[i][j];
        }
        norm_transform(i, 3) = optT.val[i][0];
    }
    
    std::cout << "Normalized transformation matrix:" << std::endl;
    std::cout << norm_transform << std::endl;
    
    // 9. Denormalize transformation
    Eigen::Matrix4f final_transform = denormalizeTransformation(
        norm_transform, source_centroid, target_centroid, max_scale);
    
    std::cout << "Final transformation matrix:" << std::endl;
    std::cout << final_transform << std::endl;
    
    // 10. Visualize result if requested
    if (visualize_after) {
        std::cout << "Visualizing registration result..." << std::endl;
        visualizeRegistrationResult(source_cloud, target_cloud, final_transform, "Registration Result");
    }
    
    return 0;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " source_cloud.pcd target_cloud.pcd [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --visualize-before, -vb     Visualize point clouds before registration" << std::endl;
    std::cout << "  --visualize-after, -va      Visualize registration result" << std::endl;
    std::cout << "  --max-points, -mp <number>  Maximum number of points in target cloud (default: 3000)" << std::endl;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "Error: Could not load point cloud file: " << filename << std::endl;
        exit(1);
    }
    
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int maxPoints) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    
    if (cloud->size() <= (size_t)maxPoints) {
        // No downsampling needed
        *downsampled = *cloud;
        return downsampled;
    }
    
    // Random sampling
    std::vector<int> indices(cloud->size());
    for (size_t i = 0; i < cloud->size(); i++) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    downsampled->resize(maxPoints);
    for (int i = 0; i < maxPoints; i++) {
        downsampled->points[i] = cloud->points[indices[i]];
    }
    downsampled->width = maxPoints;
    downsampled->height = 1;
    downsampled->is_dense = cloud->is_dense;
    
    return downsampled;
}

Eigen::Vector3f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    
    if (cloud->empty()) {
        return centroid;
    }
    
    for (const auto& point : cloud->points) {
        centroid[0] += point.x;
        centroid[1] += point.y;
        centroid[2] += point.z;
    }
    
    centroid /= static_cast<float>(cloud->points.size());
    return centroid;
}

float computeMaxDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector3f& centroid) {
    float max_dist = 0.0f;
    
    for (const auto& point : cloud->points) {
        Eigen::Vector3f pt(point.x, point.y, point.z);
        float dist = (pt - centroid).norm();
        
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    return max_dist;
}

void normalizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& source_normalized,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& target_normalized,
    Eigen::Vector3f& source_centroid,
    Eigen::Vector3f& target_centroid,
    float& max_scale) {
    
    // Compute centroids
    source_centroid = computeCentroid(source);
    target_centroid = computeCentroid(target);
    
    // Center point clouds
    source_normalized->resize(source->size());
    target_normalized->resize(target->size());
    
    // Create centered point clouds
    for (size_t i = 0; i < source->size(); i++) {
        source_normalized->points[i].x = source->points[i].x - source_centroid[0];
        source_normalized->points[i].y = source->points[i].y - source_centroid[1];
        source_normalized->points[i].z = source->points[i].z - source_centroid[2];
    }
    
    for (size_t i = 0; i < target->size(); i++) {
        target_normalized->points[i].x = target->points[i].x - target_centroid[0];
        target_normalized->points[i].y = target->points[i].y - target_centroid[1];
        target_normalized->points[i].z = target->points[i].z - target_centroid[2];
    }
    
    // Find max scale
    float source_max_dist = computeMaxDistance(source_normalized, Eigen::Vector3f::Zero());
    float target_max_dist = computeMaxDistance(target_normalized, Eigen::Vector3f::Zero());
    max_scale = std::max(source_max_dist, target_max_dist);
    
    // Scale point clouds
    for (size_t i = 0; i < source_normalized->size(); i++) {
        source_normalized->points[i].x /= max_scale;
        source_normalized->points[i].y /= max_scale;
        source_normalized->points[i].z /= max_scale;
    }
    
    for (size_t i = 0; i < target_normalized->size(); i++) {
        target_normalized->points[i].x /= max_scale;
        target_normalized->points[i].y /= max_scale;
        target_normalized->points[i].z /= max_scale;
    }
    
    source_normalized->width = source_normalized->size();
    source_normalized->height = 1;
    source_normalized->is_dense = source->is_dense;
    
    target_normalized->width = target_normalized->size();
    target_normalized->height = 1;
    target_normalized->is_dense = target->is_dense;
}

Eigen::Matrix4f denormalizeTransformation(
    const Eigen::Matrix4f& transform,
    const Eigen::Vector3f& source_centroid,
    const Eigen::Vector3f& target_centroid,
    float scale) {
    
    // Create matrices for each step
    Eigen::Matrix4f T_s = Eigen::Matrix4f::Identity();  // Translation to source centroid
    T_s.block<3, 1>(0, 3) = -source_centroid;
    
    Eigen::Matrix4f T_t = Eigen::Matrix4f::Identity();  // Translation from target centroid
    T_t.block<3, 1>(0, 3) = target_centroid;
    
    Eigen::Matrix4f S = Eigen::Matrix4f::Identity();  // Scaling
    S(0, 0) = S(1, 1) = S(2, 2) = scale;
    
    Eigen::Matrix4f S_inv = Eigen::Matrix4f::Identity();  // Inverse scaling
    S_inv(0, 0) = S_inv(1, 1) = S_inv(2, 2) = 1.0f / scale;
    
    // Combine transformations: T_t * S_inv * transform * S * T_s
    Eigen::Matrix4f denorm_transform = T_t * S_inv * transform * S * T_s;
    
    return denorm_transform;
}

std::vector<POINT3D> convertToPoint3DList(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<POINT3D> points;
    points.reserve(cloud->size());
    
    for (const auto& pt : cloud->points) {
        POINT3D p3d;
        p3d.x = pt.x;
        p3d.y = pt.y;
        p3d.z = pt.z;
        points.push_back(p3d);
    }
    
    return points;
}

void visualizePointClouds(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const std::string& title) {
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    
    // Add source point cloud in red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(source, source_color, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
    
    // Add target point cloud in green
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target");
    
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    std::cout << "Press 'q' to continue..." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizeRegistrationResult(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    const Eigen::Matrix4f& transform,
    const std::string& title) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*source, *transformed_source, transform);
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(title));
    viewer->setBackgroundColor(0, 0, 0);
    
    // Add transformed source point cloud in red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(transformed_source, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(transformed_source, source_color, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
    
    // Add target point cloud in green
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target");
    
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    std::cout << "Press 'q' to exit..." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}