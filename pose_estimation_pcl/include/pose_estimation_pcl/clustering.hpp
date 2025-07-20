#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <string>
#include <vector>

namespace pose_estimation
{

    /**
     * @brief Class for clustering point clouds and matching against models
     */
    class CloudClustering
    {
    public:
        /**
         * @brief Result structure for histogram matching
         */
        struct HistogramMatchingResult
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr best_matching_cluster;
            size_t best_matching_index = SIZE_MAX;
            float best_similarity = 0.0f;
            std::vector<float> cluster_similarities;
        };

        /**
         * @brief Structure to hold features of a cluster
         */
        struct ClusterFeatures
        {
            pcl::PointCloud<pcl::Normal>::Ptr normals;
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs;
            std::vector<float> histogram;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;

            ClusterFeatures()
            {
                normals.reset(new pcl::PointCloud<pcl::Normal>());
                fpfhs.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
                points.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
            }
        };

        /**
         * @brief Configuration for cloud clustering
         */
        struct Config
        {
            // Clustering parameters
            double cluster_tolerance = 0.02;
            int min_cluster_size = 100;
            int max_cluster_size = 25000;

            // FPFH feature parameters
            float normal_radius = 0.03f;
            float fpfh_radius = 0.05f;
            float similarity_threshold = 0.5f;
            int num_threads = 4;
            bool visualize_normals = false;

            // Super4PCS parameters
            float super4pcs_delta = 0.005f;
            float super4pcs_overlap = 0.5f;
            int super4pcs_max_iterations = 100;
        };

        /**
         * @brief Constructor
         * @param logger ROS logger
         * @param config Configuration parameters
         */
        CloudClustering(const Config &config, rclcpp::Logger logger = rclcpp::get_logger("cloud_clustering"));

        /**
         * @brief Set configuration
         * @param config New configuration parameters
         */
        void setConfig(const Config &config);

        /**
         * @brief Get current configuration
         * @return Current configuration
         */
        const Config &getConfig() const;

        /**
         * @brief Set model and compute its features
         * @param model_cloud Point cloud of the model
         */
        void setModel(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &model_cloud);

        /**
         * @brief Process a point cloud to find the best matching cluster
         * @param input_cloud Input point cloud
         * @return Point cloud of the best matching cluster, or nullptr if no match found
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr findBestCluster(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud);

        /**
         * @brief Get all clusters from the last processing
         * @return Vector of detected clusters
         */
        const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &getClusters() const;

        /**
         * @brief Get colored cloud from last clustering
         * @return Point cloud with colored clusters
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColoredClustersCloud() const;



        /**
         * @brief Compute FPFH correspondences between two clusters
         * @param model_cloud Features of the model cluster
         * @param input_cloud Features of the input cluster
         * @param correspondence_ratio Output correspondence ratio
         * @return Number of valid correspondences
         */
        void calculateFPFHCorrespondences(
            const ClusterFeatures &model_cloud,
            const ClusterFeatures &input_cloud,
            float &correspondence_ratio);

        

            
        /**
         * @brief Get model cloud
         * @return Model point cloud
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr getModelCloud() const;

        /**
         * @brief Compute similarity using Super4PCS algorithm
         * @param model_features ClusterFeatures struct of the model
         * @param cluster_features Features of the clusters to match against
         * @return Index of the best matching cluster, or -1 if no match found
         */
        int computeSuper4PCSSimilarity(
            const ClusterFeatures &model_features,
            const std::vector<ClusterFeatures> &cluster_features
        );

    private:
        /**
         * @brief Perform Euclidean clustering on input cloud
         * @param input_cloud Input point cloud
         */
        void performClustering(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud);

        /**
         * @brief Compute FPFH features for a single cloud
         * @param cloud Input point cloud
         * @param normal_radius Search radius for normals
         * @param fpfh_radius Search radius for FPFH
         * @param num_threads Number of threads to use
         * @param visualize_normals Whether to visualize normals
         * @return Computed features
         */
        ClusterFeatures computeCloudFeatures(
            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
            float normal_radius,
            float fpfh_radius,
            int num_threads,
            bool visualize_normals);

        /**
         * @brief Compute FPFH features for multiple clusters
         * @param clusters Vector of point cloud clusters
         * @return Vector of cluster features
         */
        std::vector<ClusterFeatures> computeFeatures(
            const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &clusters);

        /**
         * @brief Find best matching cluster based on FPFH histogram comparison
         * @param model_features Model features to match against
         * @param cluster_features Features of the detected clusters
         * @param similarity_threshold Minimum similarity to consider a match
         * @return HistogramMatchingResult containing the best matching cluster
         */
        HistogramMatchingResult MatchClustersByHistogram(
            const ClusterFeatures &model_features,
            const std::vector<ClusterFeatures> &cluster_features,
            float similarity_threshold);

        

        rclcpp::Logger logger_;
        Config config_;
        bool debug_time_ = false;

        // Model data
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_;
        ClusterFeatures model_features_;
        bool model_loaded_ = false;

        // Results from last processing
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr best_cluster_;
    };

} // namespace pose_estimation

#endif // CLOUD_CLUSTERING_HPP