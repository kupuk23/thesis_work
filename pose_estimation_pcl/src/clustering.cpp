#include "pose_estimation_pcl/clustering.hpp"
#include "pose_estimation_pcl/utils/pcl_utils.hpp"

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <chrono>
#include <algorithm>
#include <limits>

namespace pose_estimation {

CloudClustering::CloudClustering(const Config& config, rclcpp::Logger logger)
    : config_(config), logger_(logger)
{
    model_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    colored_clusters_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    best_cluster_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    RCLCPP_INFO(logger_, "Cloud clustering initialized");
}

void CloudClustering::setConfig(const Config& config) {
    config_ = config;
    RCLCPP_INFO(logger_, "Cloud clustering configuration updated");
}

const CloudClustering::Config& CloudClustering::getConfig() const {
    return config_;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudClustering::loadModel(const std::string& file_path) {
    
    
    // Load model point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_path, *model_cloud) == -1) {
        RCLCPP_ERROR(logger_, "Failed to load model PCD file: %s", file_path.c_str());
    } else {
        RCLCPP_INFO(logger_, "Loaded model PCD with %ld points", model_cloud->size());
        
        // Store the model and compute its features
        setModel(model_cloud);
    }
    return model_cloud;
}

void CloudClustering::setModel(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& model_cloud) {
    if (!model_cloud || model_cloud->empty()) {
        RCLCPP_ERROR(logger_, "Cannot set empty model");
        model_loaded_ = false;
        return;
    }
    
    model_cloud_ = model_cloud;
    
    // Compute features for the model
    model_features_ = computeCloudFeatures(
        model_cloud_,
        config_.normal_radius,
        config_.fpfh_radius,
        config_.num_threads,
        config_.visualize_normals
    );
    
    model_loaded_ = !model_features_.histogram.empty();
    
    if (model_loaded_) {
        RCLCPP_INFO(logger_, "Model features computed successfully");
    } else {
        RCLCPP_ERROR(logger_, "Failed to compute model features");
    }
}
CloudClustering::HistogramMatchingResult CloudClustering::MatchClustersByHistogram(
    const ClusterFeatures& model_features,
    const std::vector<ClusterFeatures>& cluster_features,
    float similarity_threshold) {
    
    HistogramMatchingResult result;
    result.best_matching_cluster.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    result.cluster_similarities.resize(cluster_features.size(), 0.0f);
    
    // Track best match information
    int best_cluster_index = -1;
    float best_similarity_score = 0.0f;
    
    // Process each cluster
    for (size_t i = 0; i < cluster_features.size(); ++i) {
        // Skip if the cluster has no features or histogram
        if (cluster_features[i].histogram.empty()) {
            RCLCPP_WARN(logger_, "Cluster %ld has no histogram, skipping", i);
            continue;
        }
        
        // Compare histograms using histogram intersection
        float similarity = 0.0f;
        for (size_t j = 0; j < std::min(model_features.histogram.size(), 
                                        cluster_features[i].histogram.size()); ++j) {
            similarity += std::min(
                model_features.histogram[j],
                cluster_features[i].histogram[j]
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
        best_cluster_index < static_cast<int>(clusters_.size())) {
        
        result.best_matching_cluster = clusters_[best_cluster_index];
        result.best_matching_index = best_cluster_index;
        result.best_similarity = best_similarity_score;
        
        RCLCPP_INFO(logger_, "Best matching cluster: %d with similarity score: %.4f", 
                  best_cluster_index, best_similarity_score);
    } else {
        RCLCPP_WARN(logger_, "Best cluster (idx: %d) similarity %.4f below threshold %.4f, rejecting match",
                  best_cluster_index, best_similarity_score, similarity_threshold);
    }
    
    return result;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudClustering::findBestCluster(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    best_cluster_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    if (!model_loaded_) {
        RCLCPP_ERROR(logger_, "No model loaded, cannot find best cluster");
        return best_cluster_;
    }
    
    if (!input_cloud || input_cloud->empty()) {
        RCLCPP_ERROR(logger_, "Input cloud is empty, cannot cluster");
        return best_cluster_;
    }
    
    // Step 1: Cluster the filtered point cloud
    RCLCPP_DEBUG(logger_, "Clustering input cloud with %ld points", input_cloud->size());
    
    // Perform clustering
    performClustering(input_cloud);
    
    if (clusters_.empty()) {
        RCLCPP_WARN(logger_, "No clusters found in input cloud");
        return best_cluster_;
    }
    
    RCLCPP_INFO(logger_, "Found %ld clusters", clusters_.size());
    
    // Step 2: Compute FPFH features for each cluster
    std::vector<ClusterFeatures> cluster_features;
    for (const auto& cluster : clusters_) {
        if (cluster->empty()) continue;
        
        ClusterFeatures features = computeCloudFeatures(
            cluster,
            config_.normal_radius,
            config_.fpfh_radius,
            config_.num_threads,
            config_.visualize_normals
        );
        
        if (!features.histogram.empty()) {
            cluster_features.push_back(features);
        }
    }
    
    if (cluster_features.empty()) {
        RCLCPP_WARN(logger_, "Failed to compute features for clusters");
        return best_cluster_;
    }
    
    // Step 3: Find best matching cluster
    auto matching_histogram_result = MatchClustersByHistogram(
        model_features_,
        cluster_features,
        config_.similarity_threshold
    );

    auto best_index = matching_histogram_result.best_matching_index;
    
    if (best_index < clusters_.size()) {
        best_cluster_ = clusters_[best_index];
        RCLCPP_INFO(logger_, "Found best matching cluster with %ld points (index: %ld)", 
                  best_cluster_->size(), best_index);
    } else {
        RCLCPP_INFO(logger_, "No cluster matched the model with sufficient similarity");
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (debug_time_) {
        RCLCPP_INFO(logger_, "Best cluster finding executed in %.3f seconds", execution_time);
    }
    
    return best_cluster_;
}

void CloudClustering::performClustering(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear previous results
    clusters_.clear();
    colored_clusters_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    try {
        // Create a KdTree for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(input_cloud);

        // Setup cluster extraction
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(config_.cluster_tolerance);
        ec.setMinClusterSize(config_.min_cluster_size);
        ec.setMaxClusterSize(config_.max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(input_cloud);
        ec.extract(cluster_indices);
        
        // Exit if no clusters found
        if (cluster_indices.empty()) {
            return;
        }
        
        // Extract each cluster and colorize it
        clusters_.reserve(cluster_indices.size());
        
        // Define some colors for visualization
        const std::vector<std::array<uint8_t, 3>> colors = pcl_utils::DEFAULT_COLORS;
        
        int j = 0;
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            // Extract cluster points
            for (const auto& idx : indices.indices) {
                cluster->push_back(input_cloud->points[idx]);
                
                // Create colored version for visualization
                pcl::PointXYZRGB colored_point = input_cloud->points[idx];
                const auto& color = colors[j % colors.size()];
                colored_point.r = color[0];
                colored_point.g = color[1];
                colored_point.b = color[2];
                colored_cluster->push_back(colored_point);
            }
            
            cluster->width = cluster->size();
            cluster->height = 1;
            cluster->is_dense = true;
            
            colored_cluster->width = colored_cluster->size();
            colored_cluster->height = 1;
            colored_cluster->is_dense = true;
            
            // Store the cluster
            clusters_.push_back(cluster);
            
            // Add to colored visualization cloud
            *colored_clusters_ += *colored_cluster;
            
            j++;
        }
        
        // Sort clusters by size (largest first)
        std::sort(clusters_.begin(), clusters_.end(),
            [](const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& a, 
               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& b) {
                return a->size() > b->size();
            });
        
        RCLCPP_INFO(logger_, "Found %zu clusters", clusters_.size());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Error during clustering: %s", e.what());
    }
    
    // Calculate execution time if debug is enabled
    if (debug_time_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end_time - start_time).count();
        RCLCPP_INFO(logger_, "Clustering executed in %.3f seconds", execution_time);
    }
}

CloudClustering::ClusterFeatures CloudClustering::computeCloudFeatures(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    float normal_radius,
    float fpfh_radius,
    int num_threads,
    bool visualize_normals) {
    
    ClusterFeatures features;
    
    // Skip if the input is empty
    if (!cloud || cloud->empty()) {
        return features;
    }
    
    try {
        // Convert XYZRGB to XYZ (normals computation doesn't need RGB)
        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        xyz_cloud->points.resize(cloud->size());
        
        for (size_t i = 0; i < cloud->size(); ++i) {
            xyz_cloud->points[i].x = cloud->points[i].x;
            xyz_cloud->points[i].y = cloud->points[i].y;
            xyz_cloud->points[i].z = cloud->points[i].z;
        }
        
        xyz_cloud->width = cloud->width;
        xyz_cloud->height = cloud->height;
        xyz_cloud->is_dense = cloud->is_dense;

        // Compute normals using PCL's Normal Estimation
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setNumberOfThreads(num_threads);
        norm_est.setInputCloud(xyz_cloud);
        norm_est.setRadiusSearch(normal_radius);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        norm_est.setSearchMethod(tree);

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        norm_est.compute(*normals);

        // Compute FPFH features
        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setNumberOfThreads(num_threads);
        fpfh.setInputCloud(xyz_cloud);
        fpfh.setInputNormals(normals);
        fpfh.setSearchMethod(tree);
        fpfh.setRadiusSearch(fpfh_radius);
        
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfh.compute(*fpfhs);

        // Calculate average FPFH histogram with proper normalization
        std::vector<float> avg_histogram(33, 0.0f);

        for (const auto& fpfh_point : fpfhs->points) {
            // First normalize this individual histogram
            float point_sum = 0.0f;
            for (int i = 0; i < 33; i++) {
                point_sum += fpfh_point.histogram[i];
            }
            
            if (point_sum > 0) {  // Avoid division by zero
                for (int i = 0; i < 33; i++) {
                    // Add normalized value to average
                    avg_histogram[i] += fpfh_point.histogram[i] / point_sum;
                }
            }
        }

        // Finalize the average
        if (!fpfhs->empty()) {
            for (int i = 0; i < 33; i++) {
                avg_histogram[i] /= static_cast<float>(fpfhs->size());
            }
        }

        // Fill the result structure
        features.normals = normals;
        features.fpfhs = fpfhs;
        features.histogram = avg_histogram;
        features.points = cloud;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Error computing features: %s", e.what());
    }
    
    return features;
}

std::vector<CloudClustering::ClusterFeatures> CloudClustering::computeFeatures(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters) {
    
    std::vector<ClusterFeatures> features_vec;
    features_vec.reserve(clusters.size());
    
    for (const auto& cluster : clusters) {
        ClusterFeatures features = computeCloudFeatures(
            cluster,
            config_.normal_radius,
            config_.fpfh_radius,
            config_.num_threads,
            config_.visualize_normals
        );
        
        if (!features.histogram.empty()) {
            features_vec.push_back(features);
        }
    }
    
    return features_vec;
}


const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& CloudClustering::getClusters() const {
    return clusters_;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudClustering::getColoredClustersCloud() const {
    return colored_clusters_;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudClustering::getModelCloud() const {
    return model_cloud_;
}

} // namespace pose_estimation