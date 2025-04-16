#include "pose_estimation_pcl/go_icp_wrapper.hpp"
#include <iostream>
#include <algorithm>
#include <random>

namespace go_icp {

GoICPWrapper::GoICPWrapper() 
    : last_error_(0.0f), last_registration_time_(0.0f), last_normalization_scale_(1.0f) {}

Eigen::Matrix4f GoICPWrapper::registerPointClouds(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& model_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scene_cloud,
    int max_scene_points,
    bool debug,
    int dt_size,
    float dt_expand_factor,
    float mse_thresh) {
    
    auto start = std::chrono::high_resolution_clock::now();

    if (debug) {
        std::cout << "Starting Go-ICP registration" << std::endl;
        std::cout << "Model points: " << model_cloud->size() << ", Scene points: " << scene_cloud->size() << std::endl;
    }

    // 1. Convert RGB clouds to XYZ (Go-ICP doesn't need color information)
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_xyz = convertRGBtoXYZ(model_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_xyz = convertRGBtoXYZ(scene_cloud);

    // 2. Downsample scene cloud if necessary (for performance)
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_downsampled = downsamplePointCloud(scene_xyz, max_scene_points);
    
    if (debug) {
        std::cout << "After downsampling - Scene points: " << scene_downsampled->size() << std::endl;
    }

    // 3. Normalize point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_normalized(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_normalized(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3f model_centroid, scene_centroid;
    float max_scale;

    normalizePointCloud(scene_downsampled, model_xyz, scene_normalized, model_normalized,
        scene_centroid, model_centroid, max_scale);
    last_normalization_scale_ = max_scale;

    if (debug) {
        std::cout << "Normalized point clouds with scale factor: " << max_scale << std::endl;
    }

    // 4. Convert to Go-ICP format
    std::vector<POINT3D> model_points = convertToPoint3DList(model_normalized);
    std::vector<POINT3D> scene_points = convertToPoint3DList(scene_normalized);

    // 5. Initialize Go-ICP
    GoICP goicp;
    ROTNODE rNode;
    TRANSNODE tNode;
    
    // Set rotation search space (full rotation space)
    rNode.a = -PI;
    rNode.b = -PI;
    rNode.c = -PI;
    rNode.w = 2*PI;
    
    // Set translation search space
    tNode.x = -0.5;
    tNode.y = -0.5;
    tNode.z = -0.5;
    tNode.w = 1.0;
    
    // Set Go-ICP parameters
    goicp.MSEThresh = mse_thresh;  // Mean Square Error threshold
    goicp.trimFraction = 0.0;  // Trimming fraction (0.0 = no trimming)
    goicp.doTrim = (goicp.trimFraction >= 0.001);
    
    // Prepare data for Go-ICP
    goicp.pModel = model_points.data();
    goicp.Nm = model_points.size();
    goicp.pData = scene_points.data();
    goicp.Nd = scene_points.size();
    
    // Set DT parameters
    goicp.dt.SIZE = dt_size;
    goicp.dt.expandFactor = dt_expand_factor;
    
    // Set initial rotation and translation nodes
    goicp.initNodeRot = rNode;
    goicp.initNodeTrans = tNode;
    
    
    
    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    goicp.BuildDT();


    // Calculate the time taken to build the distance transform
    
    // 6. Run Go-ICP
    if (debug) {
        auto end_time = std::chrono::high_resolution_clock::now();

        auto dt_time = std::chrono::duration<float>(end_time - start_time).count();
    
        std::cout << "Building Distance Transform... (took "<< dt_time << " ms"  << std::endl;
    }

    if (debug) {
        std::cout << "Starting Go-ICP registration..." << std::endl;
    }
    float error = goicp.Register();
    last_error_ = error;
    
    if (debug) {
        std::cout << "Registration completed with error: " << error << std::endl;
    }
    
    // 7. Get optimal transformation
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
    
    // 8. Denormalize transformation
    Eigen::Matrix4f final_transform = denormalizeTransformation(
        norm_transform, scene_centroid, model_centroid, max_scale);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_registration_time_ = std::chrono::duration<float>(end - start).count();
    
    if (debug) {
        std::cout << "Go-ICP registration time: " << last_registration_time_ << " seconds" << std::endl;
    }
    
    return final_transform;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GoICPWrapper::convertRGBtoXYZ(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    xyz_cloud->points.resize(rgb_cloud->points.size());
    xyz_cloud->width = rgb_cloud->width;
    xyz_cloud->height = rgb_cloud->height;
    xyz_cloud->is_dense = rgb_cloud->is_dense;
    
    for (size_t i = 0; i < rgb_cloud->points.size(); ++i) {
        xyz_cloud->points[i].x = rgb_cloud->points[i].x;
        xyz_cloud->points[i].y = rgb_cloud->points[i].y;
        xyz_cloud->points[i].z = rgb_cloud->points[i].z;
    }
    
    return xyz_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GoICPWrapper::downsamplePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int max_points) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    
    if (cloud->size() <= (size_t)max_points) {
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
    
    downsampled->resize(max_points);
    for (int i = 0; i < max_points; i++) {
        downsampled->points[i] = cloud->points[indices[i]];
    }
    downsampled->width = max_points;
    downsampled->height = 1;
    downsampled->is_dense = cloud->is_dense;
    
    return downsampled;
}

Eigen::Vector3f GoICPWrapper::computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

float GoICPWrapper::computeMaxDistance(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Vector3f& centroid) {
    
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

void GoICPWrapper::normalizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& data_pc,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& model_pc,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& data_normalized,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& model_normalized,
    Eigen::Vector3f& data_centroid,
    Eigen::Vector3f& model_centroid,
    float& max_scale) {
    
    // Compute centroids
    data_centroid = computeCentroid(data_pc);
    model_centroid = computeCentroid(model_pc);
    
    // Center point clouds
    data_normalized->resize(data_pc->size());
    model_normalized->resize(model_pc->size());
    
    // Create centered point clouds
    for (size_t i = 0; i < data_pc->size(); i++) {
        data_normalized->points[i].x = data_pc->points[i].x - data_centroid[0];
        data_normalized->points[i].y = data_pc->points[i].y - data_centroid[1];
        data_normalized->points[i].z = data_pc->points[i].z - data_centroid[2];
    }
    
    for (size_t i = 0; i < model_pc->size(); i++) {
        model_normalized->points[i].x = model_pc->points[i].x - model_centroid[0];
        model_normalized->points[i].y = model_pc->points[i].y - model_centroid[1];
        model_normalized->points[i].z = model_pc->points[i].z - model_centroid[2];
    }
    
    // Find max scale
    float source_max_dist = computeMaxDistance(data_normalized, Eigen::Vector3f::Zero());
    float target_max_dist = computeMaxDistance(model_normalized, Eigen::Vector3f::Zero());
    max_scale = std::max(source_max_dist, target_max_dist);
    
    // Scale point clouds
    for (size_t i = 0; i < data_normalized->size(); i++) {
        data_normalized->points[i].x /= max_scale;
        data_normalized->points[i].y /= max_scale;
        data_normalized->points[i].z /= max_scale;
    }
    
    for (size_t i = 0; i < model_normalized->size(); i++) {
        model_normalized->points[i].x /= max_scale;
        model_normalized->points[i].y /= max_scale;
        model_normalized->points[i].z /= max_scale;
    }
    
    data_normalized->width = data_normalized->size();
    data_normalized->height = 1;
    data_normalized->is_dense = data_pc->is_dense;
    
    model_normalized->width = model_normalized->size();
    model_normalized->height = 1;
    model_normalized->is_dense = model_pc->is_dense;
}

Eigen::Matrix4f GoICPWrapper::denormalizeTransformation(
    const Eigen::Matrix4f& transform,
    const Eigen::Vector3f& data_centroid,
    const Eigen::Vector3f& model_centroid,
    float scale) {
    
    // Create matrices for each step
    Eigen::Matrix4f T_data = Eigen::Matrix4f::Identity();  // Translation to source centroid
    T_data.block<3, 1>(0, 3) = data_centroid;
    
    Eigen::Matrix4f T_model = Eigen::Matrix4f::Identity();  // Translation from target centroid
    T_model.block<3, 1>(0, 3) = -model_centroid;
    
    Eigen::Matrix4f S = Eigen::Matrix4f::Identity();  // Scaling
    S(0, 0) = S(1, 1) = S(2, 2) = scale;
    
    Eigen::Matrix4f S_inv = Eigen::Matrix4f::Identity();  // Inverse scaling
    S_inv(0, 0) = S_inv(1, 1) = S_inv(2, 2) = 1.0f / scale;
    
    // Combine transformations: T_model * S_inv * transform * S * T_data
    Eigen::Matrix4f denorm_transform = T_data * S_inv * transform * S * T_model;
    
    return denorm_transform;
}

std::vector<POINT3D> GoICPWrapper::convertToPoint3DList(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    
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

} // namespace go_icp