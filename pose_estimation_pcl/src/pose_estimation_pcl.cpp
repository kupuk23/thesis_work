#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_events_filter.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <cfloat>  // For FLT_MAX
#include <thread> 

// include tf2 buffer for lookup transform
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include "pose_estimation_pcl/pcl_utils.hpp"    // Our utility header
#include "pose_estimation_pcl/ros_utils.hpp"    // Our ROS utility header
#include "pose_estimation_pcl/go_icp_wrapper.hpp"  // Our Go-ICP wrapper

using namespace std::chrono_literals;


class PoseEstimationPCL : public rclcpp::Node {
public:
    PoseEstimationPCL() : Node("pose_estimation_pcl") {
        // Initialize parameters
        // General parameters
        processing_period_ms_ = this->declare_parameter("general.processing_period_ms", 100);
        goicp_debug_ = this->declare_parameter("general.goicp_debug", false);
        save_debug_clouds_ = this->declare_parameter("general.save_debug_clouds", false);
        voxel_size_ = this->declare_parameter("general.voxel_size", 0.05);
        object_frame_ = this->declare_parameter("general.object_frame", "grapple");
        save_to_pcd_ = this->declare_parameter("general.save_to_pcd", false);
        suffix_name_pcd_ = this->declare_parameter("general.suffix_name_pcd", "test");

        // Gen ICP params
        gicp_fitness_threshold_ = this->declare_parameter("gen_icp.fitness_threshold", 0.05);
        gicp_max_iterations_ = this->declare_parameter("gen_icp.max_iterations", 100);
        gicp_transformation_epsilon_ = this->declare_parameter("gen_icp.transformation_epsilon", 1e-6);
        gicp_max_correspondence_distance_ = this->declare_parameter("gen_icp.max_correspondence_distance", 0.1);
        gicp_euclidean_fitness_epsilon_ = this->declare_parameter("gen_icp.euclidean_fitness_epsilon", 5e-5);
        gicp_ransac_outlier_threshold_ = this->declare_parameter("gen_icp.ransac_threshold", 0.05);

        // GO ICP params
        use_goicp_ = this->declare_parameter("go_icp.use_goicp", false);
        goicp_mse_thresh_ = this->declare_parameter("go_icp.mse_threshold", 0.001);
        goicp_dt_size_ = this->declare_parameter("go_icp.dt_size", 25);
        goicp_expand_factor_ = this->declare_parameter("go_icp.dt_expandFactor", 4.0);
        
        
        // Clustering parameters
        cluster_tolerance_ = this->declare_parameter("clustering.cluster_tolerance", 0.02);
        min_cluster_size_ = this->declare_parameter("clustering.min_cluster_size", 100);
        max_cluster_size_ = this->declare_parameter("clustering.max_cluster_size", 25000);

        
        // 3D Descriptor parameters
        visualize_normals_ = this->declare_parameter<bool>("3d_decriptors.visualize_normals", false);
        normal_radius_ = this->declare_parameter<double>("3d_decriptors.normal_radius", 0.03);
        fpfh_radius_ = this->declare_parameter<double>("3d_decriptors.fpfh_radius", 0.05);
        similarity_threshold_ = this->declare_parameter<double>("3d_decriptors.similarity_threshold", 0.6);

        // Plane segmentation parameters
        plane_distance_threshold_ = this->declare_parameter("plane_detection.plane_distance_threshold", 0.01);
        max_plane_iterations_ = this->declare_parameter("plane_detection.max_plane_iterations", 100);
        min_plane_points_ = this->declare_parameter("plane_detection.min_plane_points", 800);
        max_planes_ = this->declare_parameter("plane_detection.max_planes", 3);

        array_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("array_topic", 10);

        // Initialize Go-ICP wrapper
        goicp_wrapper_ = std::make_unique<go_icp::GoICPWrapper>();
    
        // Initialize TF2 buffer and listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    
        // Create callback groups for concurrent execution
        callback_group_subscription_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callback_group_processing_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        
        // Set up the parameter callback
        parameter_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&PoseEstimationPCL::parametersCallback, this, std::placeholders::_1));
        
            
        // Configure subscription options with callback group
        auto subscription_options = rclcpp::SubscriptionOptions();
        subscription_options.callback_group = callback_group_subscription_;
        
        // Initialize publishers and subscribers
        pointcloud_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/points", 10, 
            std::bind(&PoseEstimationPCL::pointcloud_callback, this, std::placeholders::_1),
            subscription_options);
        
        icp_result_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose/icp_result", 10);
        
        
        
        // Create publishers for Go-ICP results
        goicp_result_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose/goicp_result", 10);

        // Create a publisher for point clouds debuggin
        cloud_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ds_pc", 1);
        plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/planes_pc", 1);
        filtered_plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_plane_pc", 1);
        clustered_plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/clustered_plane_pc", 1);
        pre_processed_debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pre_processed_pc", 1);

        array_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "/pose_estimation/cluster_similarities", 10);
        

        
        // Load model cloud
        if (object_frame_ == "grapple") 
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/grapple_fixture_v2.pcd", this->get_logger());
        else if (object_frame_ == "handrail")
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", this->get_logger());
        else if (object_frame_ == "docking_st")
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/astrobee_dock_ds.pcd", this->get_logger());
        
        model_vector = {model_cloud_};

        // Initialize processing timer in separate callback group
        processing_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(processing_period_ms_),
            std::bind(&PoseEstimationPCL::process_data, this),
            callback_group_processing_);
        
        // Initialize flags and mutex
        has_new_data_ = false;
        tracking_initialized_ = false;  // Start with no tracking to force Go-ICP on first frame
        
        RCLCPP_INFO(this->get_logger(), "PoseEstimationPCL node initialized with Go-ICP integration");
    }

private:

    // Callback for receiving point cloud data (lightweight)
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg) {
        try {
            // Store the pointcloud and perform lightweight operations
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            latest_cloud_msg_ = pointcloud_msg;
            
                       
            // Update flag to indicate new data is available
            has_new_data_ = true;

            
            
            RCLCPP_DEBUG(this->get_logger(), "Received new point cloud data");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in pointcloud callback: %s", e.what());
        }
    }
    
    // Run GICP registration with a given initial transformation
    bool run_gicp(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& source_cloud,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target_cloud,
        const Eigen::Matrix4f& initial_transform,
        Eigen::Matrix4f& result_transform,
        float& fitness_score) {
        
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> gicp;
        
        gicp.setInputSource(source_cloud);
        gicp.setInputTarget(target_cloud);
        gicp.setUseReciprocalCorrespondences(true);
        
        // Set GICP parameters
        gicp.setMaximumIterations(gicp_max_iterations_);
        gicp.setTransformationEpsilon(gicp_transformation_epsilon_);
        gicp.setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
        gicp.setEuclideanFitnessEpsilon(gicp_euclidean_fitness_epsilon_);
        gicp.setRANSACOutlierRejectionThreshold(gicp_ransac_outlier_threshold_);
        
        // Align using provided initial transform
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        gicp.align(*aligned_cloud, initial_transform);
        
        // Get results
        bool converged = gicp.hasConverged();
        fitness_score = gicp.getFitnessScore();
        result_transform = gicp.getFinalTransformation();
        
        return converged;
    }
    
    // Process point cloud data
void process_data() {
    // Check if we have new data to process
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
    
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!has_new_data_ || !latest_cloud_msg_) {
            return;  // No new data to process
        }
        
        // Copy the data we need for processing
        cloud_msg = latest_cloud_msg_;
        has_new_data_ = false;  // Reset flag
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert and preprocess point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pcl_utils::convertPointCloud2ToPCL(cloud_msg);
        auto preprocessed_cloud = preprocess_pointcloud(cloud, cloud_msg);
        
        if (save_to_pcd_) {
            // Save the point cloud to PCD file for debugging
            std::string filename = "/home/tafarrel/o3d_logs/" + object_frame_ + suffix_name_pcd_ + ".pcd";
            pcl_utils::saveToPCD(preprocessed_cloud, filename, this->get_logger());
        }

        // Check if cloud is empty
        if (preprocessed_cloud->size() < 100) {
            RCLCPP_INFO(this->get_logger(), "Scene point cloud is empty, skipping registration");
            ros_utils::publish_empty_pose(icp_result_publisher_ ,cloud_msg);
            tracking_initialized_ = false;
            return;
        }
        

        // Perform registration
        Eigen::Matrix4f final_transformation = performRegistration(preprocessed_cloud, cloud_msg);
        
        // Publish and broadcast results
        ros_utils::publish_registration_results(final_transformation, cloud_msg, 
            icp_result_publisher_, tf_broadcaster_, object_frame_, "_icp");
        
        
        auto end_time = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), 
            "Total processing time: %.3f s",
            std::chrono::duration<double>(end_time - start_time).count());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
        tracking_initialized_ = false;  // Reset tracking on error
    }
}

// Perform registration (Go-ICP, GICP, or both)
Eigen::Matrix4f performRegistration(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& preprocessed_cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    // initialize final_transformation with identity matrix 4x4
    Eigen::Matrix4f final_transformation;
    final_transformation = Eigen::Matrix4f::Identity();

    float gicp_score = 0.0f;
    
    // Determine if we need to use Go-ICP
    // only run Go-ICP if:
    // 1. use_goicp_ is true
    // 2. tracking is not initialized or previous GICP score is above threshold
    // 3. object is detected
    
    bool run_goicp = use_goicp_ && (!tracking_initialized_ || previous_gicp_score_ > gicp_fitness_threshold_) && object_detected_;
    
    // First try GICP with previous transform if we have good tracking
    if (!run_goicp && tracking_initialized_) {
        RCLCPP_INFO(this->get_logger(), "Using GICP with previous transformation");
        
        bool gen_icp_converged = run_gicp(
            model_cloud_, 
            preprocessed_cloud, 
            previous_transformation_, 
            final_transformation, 
            gicp_score
        );
        
        if (gen_icp_converged && gicp_score < gicp_fitness_threshold_) {
            RCLCPP_INFO(this->get_logger(), "GICP converged successfully with score: %f", gicp_score);
            previous_gicp_score_ = gicp_score;
            previous_transformation_ = final_transformation;
        } else {
            RCLCPP_WARN(this->get_logger(), 
                "GICP %s with poor score: %f, falling back to Go-ICP", 
                gen_icp_converged ? "converged" : "failed to converge", 
                gicp_score);
            run_goicp = true;
        }
    }
    
    // Run Go-ICP if needed (first frame or GICP failed)
    // TODO: Debug GO-ICP transformation result, orientation check
    if (run_goicp) {
        final_transformation = run_go_ICP(preprocessed_cloud, cloud_msg);
    }
    
    return final_transformation;
}

// Run Go-ICP registration and GICP refinement
Eigen::Matrix4f run_go_ICP(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& preprocessed_cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    RCLCPP_INFO(this->get_logger(), "Using Go-ICP for global alignment");
    
    // Run Go-ICP registration
    Eigen::Matrix4f alignment_transform = goicp_wrapper_->registerPointClouds(
        model_cloud_,
        preprocessed_cloud,
        2000,  // Max target points 
        goicp_debug_,
        goicp_dt_size_,
        goicp_expand_factor_,
        goicp_mse_thresh_
        
    );

    
    // Get Go-ICP statistics
    float goicp_error = goicp_wrapper_->getLastError();
    float goicp_time = goicp_wrapper_->getLastRegistrationTime();

    /// print the rotation matrix
    Eigen::Matrix3f rotation_matrix = alignment_transform.block<3, 3>(0, 0);
    cout << "Rotation matrix: " << endl << rotation_matrix << endl;
    RCLCPP_INFO(this->get_logger(), 
        "Go-ICP completed in %.3f seconds with error: %.5f", 
        goicp_time, goicp_error);

    
    // Publish Go-ICP result
    ros_utils::publish_registration_results(alignment_transform, cloud_msg, 
        goicp_result_publisher_, 
        tf_broadcaster_, 
        object_frame_, "_goicp");

    
    
    // Now refine with GICP using Go-ICP result as initial guess
    Eigen::Matrix4f final_transformation;
    float gicp_score = 0.0f;

    //TODO: Perform GOCIP orientation check, if orientation is not correct, rerun go-ICP until satisfied
    
    bool gen_icp_converged = run_gicp(
        model_cloud_, 
        preprocessed_cloud, 
        alignment_transform, 
        final_transformation, 
        gicp_score
    );
    
    if (gen_icp_converged) {
        RCLCPP_INFO(this->get_logger(), "GICP refinement converged with score: %f", gicp_score);
        // extract rotation matrix from final_transformation
        Eigen::Matrix3f rotation_matrix = final_transformation.block<3, 3>(0, 0);
        cout << "Final rotation matrix: " << endl << rotation_matrix << endl;

    } else {
        RCLCPP_WARN(this->get_logger(), "GICP refinement failed, using Go-ICP result directly");
        final_transformation = alignment_transform;
        gicp_score = goicp_error;  // Use Go-ICP error as score
    }
    
    // Update tracking state
    tracking_initialized_ = true;
    previous_gicp_score_ = gicp_score;
    previous_transformation_ = final_transformation;
    
    return final_transformation;
}


    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr preprocess_pointcloud(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud, 
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // Downsample using voxel grid
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setInputCloud(input_cloud);
        voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        voxel_grid.filter(*filtered_cloud);
        
        // Filter by Z and X (matching the Python implementation)
        pcl::PassThrough<pcl::PointXYZRGB> pass_z;
        pass_z.setInputCloud(filtered_cloud);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(-0.7, FLT_MAX);  // Z > -0.7
        pass_z.filter(*filtered_cloud);
        
        pcl::PassThrough<pcl::PointXYZRGB> pass_x;
        pass_x.setInputCloud(filtered_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(-FLT_MAX, 2.0);  // X < 2
        pass_x.filter(*filtered_cloud);

        // remove largest plane (wall)
        auto segmentation_result = pcl_utils::detect_and_remove_planes(filtered_cloud, this->get_logger(), true);

        // cluster the pointclouds
        auto clustering_result = pcl_utils::cluster_point_cloud(segmentation_result.remaining_cloud, cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        
        
        auto cluster_features = pcl_utils::computeFPFHFeatures(clustering_result.individual_clusters,
            normal_radius_,  // normal radius - adjust based on your point cloud density
            fpfh_radius_,  // feature radius - adjust based on your point cloud density
            0,     // auto-detect thread count
            visualize_normals_
        );        

        auto model_features_vector = pcl_utils::computeFPFHFeatures(model_vector, normal_radius_, fpfh_radius_, 0, visualize_normals_);
        
        if (!model_features_vector.empty()) {
            model_features = model_features_vector[0];
            RCLCPP_INFO(this->get_logger(), "Model FPFH features computed successfully");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute model FPFH features");
        }
        
        
        auto matching_result = pcl_utils::findBestClusterByHistogram(
            model_features,
            cluster_features,
            clustering_result.individual_clusters,
            similarity_threshold_,
            this->get_logger()
        );

        // if result.best_matching_cluster is nullptr, return filtered_cloud
        if (matching_result.best_matching_cluster == nullptr) {
            RCLCPP_WARN(this->get_logger(), "No matching cluster found, returning filtered cloud");
            object_detected_ = false;
            return filtered_cloud;
        }

        // use the index from cluster_similarity with highest value, and return the cluster pointcloud


        // publish the similarity results to the array topic
        ros_utils::publish_array(array_publisher_, matching_result.cluster_similarities);

        // If you want to publish the planes cloud:
        if (save_debug_clouds_ && segmentation_result.planes_cloud->size() > 0) {
            // publish the downsampled pointcloud
            ros_utils::publish_debug_cloud(filtered_cloud, cloud_msg, cloud_debug_pub_, save_debug_clouds_);
        

            // publish the detected planes
            ros_utils::publish_debug_cloud(segmentation_result.planes_cloud, cloud_msg, plane_debug_pub_, save_debug_clouds_);
            // publish the filtered planes
            ros_utils::publish_debug_cloud(segmentation_result.remaining_cloud, cloud_msg, filtered_plane_debug_pub_, save_debug_clouds_);
            // publish the clustered planes
            ros_utils::publish_debug_cloud(clustering_result.colored_cloud, cloud_msg, clustered_plane_debug_pub_, save_debug_clouds_);

            ros_utils::publish_debug_cloud(matching_result.best_matching_cluster, cloud_msg, pre_processed_debug_pub, save_debug_clouds_);
        }
        object_detected_ = true;
        
        // return filtered_cloud;
        return matching_result.best_matching_cluster;
    }

    
    // Parameter callback function to handle dynamic updates
    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter>& parameters) {
        
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        
        
    for (const auto & param : parameters) {

        if (param.get_name() == "general.processing_period_ms") {
            processing_period_ms_ = param.as_int();
            // Reset timer if needed
            processing_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(processing_period_ms_),
                std::bind(&PoseEstimationPCL::process_data, this),
                callback_group_processing_);
        } else if (param.get_name() == "general.save_to_pcd") {
            save_to_pcd_ = param.as_bool();
        } else if (param.get_name() == "general.goicp_debug") {
            goicp_debug_ = param.as_bool();
        } else if (param.get_name() == "general.save_debug_clouds") {
            save_debug_clouds_ = param.as_bool();
        } else if (param.get_name() == "general.voxel_size") {
            voxel_size_ = param.as_double();
        } else if (param.get_name() == "general.object_frame") {
            object_frame_ = param.as_string();
            if (object_frame_ == "grapple") 
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/grapple_fixture_v2.pcd", this->get_logger());
            else if (object_frame_ == "handrail")
                model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", this->get_logger());
            else if (object_frame_ == "docking_st")
                model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/astrobee_dock_ds.pcd", this->get_logger());
            
            model_vector = {model_cloud_};
        } else if (param.get_name() == "general.suffix_name_pcd") {
            suffix_name_pcd_ = param.as_string();

        } else if (param.get_name() == "go_icp.use_goicp") {
            use_goicp_ = param.as_bool();
        } else if (param.get_name() == "go_icp.mse_threshold") {
            goicp_mse_thresh_ = param.as_double();
        } else if (param.get_name() == "go_icp.dt_size") {
            goicp_dt_size_ = param.as_int();
        } else if (param.get_name() == "go_icp.dt_expandFactor") {
            goicp_expand_factor_ = param.as_double();
            
        } else if (param.get_name() == "gen_icp.fitness_threshold") {
            gicp_fitness_threshold_ = param.as_double();
        } else if (param.get_name() == "gen_icp.max_iterations") {
            gicp_max_iterations_ = param.as_int();
        } else if (param.get_name() == "gen_icp.transformation_epsilon") {
            gicp_transformation_epsilon_ = param.as_double();
        } else if (param.get_name() == "gen_icp.max_correspondence_distance") {
            gicp_max_correspondence_distance_ = param.as_double();
        } else if (param.get_name() == "gen_icp.euclidean_fitness_epsilon") {
            gicp_euclidean_fitness_epsilon_ = param.as_double();
        } else if (param.get_name() == "gen_icp.ransac_threshold") {
            gicp_ransac_outlier_threshold_ = param.as_double();


        } else if (param.get_name() == "clustering.cluster_tolerance") {
            cluster_tolerance_ = param.as_double();
        } else if (param.get_name() == "clustering.min_cluster_size") {
            min_cluster_size_ = param.as_int();
        } else if (param.get_name() == "clustering.max_cluster_size") {
            max_cluster_size_ = param.as_int();

        } else if (param.get_name() == "3d_decriptors.visualize_normals") {
            visualize_normals_ = param.as_bool();
        } else if (param.get_name() == "3d_decriptors.normal_radius") {
            normal_radius_ = param.as_double();
        } else if (param.get_name() == "3d_decriptors.fpfh_radius") {
            fpfh_radius_ = param.as_double();
        } else if (param.get_name() == "3d_decriptors.similarity_threshold") {
            similarity_threshold_ = param.as_double();
            
        } else if (param.get_name() == "plane_detection.plane_distance_threshold") {
            plane_distance_threshold_ = param.as_double();
        } else if (param.get_name() == "plane_detection.max_plane_iterations") {
            max_plane_iterations_ = param.as_int();
        } else if (param.get_name() == "plane_detection.min_plane_points") {
            min_plane_points_ = param.as_int();
        } else if (param.get_name() == "plane_detection.max_planes") {
            max_planes_ = param.as_int();

        } else {
                RCLCPP_WARN(this->get_logger(), "Unknown parameter: %s", param.get_name().c_str());
            }
        }
        
        return result;
    }

    // Node members
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr icp_result_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goicp_result_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_debug_pub_;
    rclcpp::TimerBase::SharedPtr processing_timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_plane_debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clustered_plane_debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pre_processed_debug_pub;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr array_publisher_;
    
    
    // Callback groups for concurrent execution
    rclcpp::CallbackGroup::SharedPtr callback_group_subscription_;
    rclcpp::CallbackGroup::SharedPtr callback_group_processing_;
    OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

    
    // General Parameters
    double voxel_size_;
    bool save_debug_clouds_;
    std::string object_frame_;
    int processing_period_ms_;
    bool save_to_pcd_;
    std::string suffix_name_pcd_;

    // Go-ICP parameters
    bool goicp_debug_;
    bool use_goicp_;
    double goicp_mse_thresh_;
    int goicp_dt_size_;
    double goicp_expand_factor_;
    bool object_detected_ = false;

    // GICP parameters
    double gicp_fitness_threshold_;
    double gicp_max_iterations_;
    double gicp_transformation_epsilon_;
    double gicp_max_correspondence_distance_;
    double gicp_euclidean_fitness_epsilon_;
    double gicp_ransac_outlier_threshold_;

    // 3D descriptor parameters 
    pcl_utils::ClusterFeatures model_features;
    
    //clustering parameters
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    bool visualize_normals_;
    float normal_radius_;
    float fpfh_radius_;
    double similarity_threshold_;

    // Plane segmentation parameters 
    double plane_distance_threshold_;
    int max_plane_iterations_;
    int min_plane_points_;
    int max_planes_;
    bool remove_planes_;
    
    // TF2 buffer and listener
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> model_vector;
    

    // Go-ICP wrapper
    std::unique_ptr<go_icp::GoICPWrapper> goicp_wrapper_;
    
    // Data storage with thread synchronization
    std::mutex data_mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_msg_;
    
    // Eigen::Matrix4f latest_initial_transform_;
    bool has_new_data_;
    
    // Tracking state
    bool tracking_initialized_;
    float previous_gicp_score_ = std::numeric_limits<float>::max();
    Eigen::Matrix4f previous_transformation_ = Eigen::Matrix4f::Identity();
    

    
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    // Create executor that can handle multiple callback groups
    rclcpp::executors::MultiThreadedExecutor executor;
    
    auto node = std::make_shared<PoseEstimationPCL>();
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}