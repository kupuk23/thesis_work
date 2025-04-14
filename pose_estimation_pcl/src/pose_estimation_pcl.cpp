#include <rclcpp/rclcpp.hpp>
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

struct PlaneInfo {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    size_t num_points;
    pcl::ModelCoefficients::Ptr coefficients;
    int index;  // To track the original detection order
};

static const std::vector<std::array<uint8_t, 3>> DEFAULT_COLORS = {
    {255, 0, 0},      // Red
    {0, 255, 0},      // Green
    {0, 0, 255},      // Blue
    {255, 255, 0},    // Yellow
    {255, 0, 255},    // Magenta
    {0, 255, 255},    // Cyan
    {255, 128, 0},    // Orange
    {128, 0, 255},    // Purple
    {0, 128, 255},    // Light Blue
    {255, 0, 128}     // Pink
};

class PoseEstimationPCL : public rclcpp::Node {
public:
    PoseEstimationPCL() : Node("pose_estimation_pcl") {
        // Initialize parameters
        // General parameters
        voxel_size_ = this->declare_parameter<double>("voxel_size", 0.05);  // Default voxel size
        save_debug_clouds_ = this->declare_parameter<bool>("save_debug_clouds", false);
        object_frame_ = this->declare_parameter<std::string>("object_frame", "grapple");
        processing_period_ms_ = this->declare_parameter<int>("processing_period_ms", 100);  // 10 Hz default
        use_goicp_ = this->declare_parameter<bool>("use_goicp", false);
        goicp_debug_ = this->declare_parameter<bool>("goicp_debug", false);


        // Clustering parameters
        cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.02);
        min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 100);
        max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 25000);
        // Plane segmentation parameters
        plane_distance_threshold_ = this->declare_parameter<double>("plane_distance_threshold", 0.01);
        max_plane_iterations_ = this->declare_parameter<int>("max_plane_iterations", 100);
        min_plane_points_ = this->declare_parameter<int>("min_plane_points", 800);
        max_planes_ = this->declare_parameter<int>("max_planes", 3);
        
        // Registration error thresholds
        gicp_fitness_threshold_ = this->declare_parameter<double>("gicp_fitness_threshold", 0.05);
        
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
        
        // Create a publisher for debugged point clouds if needed
        cloud_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/debug_pointcloud", 1);
        
        // Create publishers for Go-ICP results
        goicp_result_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose/goicp_result", 10);

        // Create a publisher for plane debug point clouds
        plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_plane_pointcloud", 1);
        filtered_plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_filtered_plane_pointcloud", 1);
        clustered_plane_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_clustered_plane_pointcloud", 1);

        
        // Load model cloud
        if (object_frame_ == "grapple") 
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/grapple_fixture_v2.pcd", this->get_logger());
        else if (object_frame_ == "handrail")
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", this->get_logger());
        else if (object_frame_ == "docking_st")
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/astrobee_dock_ds.pcd", this->get_logger());
        

        // Initialize processing timer in separate callback group
        processing_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(processing_period_ms_),
            std::bind(&PoseEstimationPCL::process_data, this),
            callback_group_processing_);
            
        // Initialize empty pose
        empty_pose_.header.frame_id = "map";  // Default frame
        
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
            
            // Get the initial transformation from the object pose using TF2 lookup
            // Note: We'll keep this for fallback purposes, but it will be ignored if Go-ICP is used
            // latest_initial_transform_ = pcl_utils::transform_obj_pose(
            //     pointcloud_msg, 
            //     *tf_buffer_, 
            //     object_frame_,
            //     this->get_logger()
            // );
            
            // Update flag to indicate new data is available
            has_new_data_ = true;
            
            RCLCPP_DEBUG(this->get_logger(), "Received new point cloud data");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in pointcloud callback: %s", e.what());
        }
    }
    
    // Run GICP registration with a given initial transformation
    bool runGICP(
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
        gicp.setMaximumIterations(100);
        gicp.setTransformationEpsilon(1e-6);
        gicp.setMaxCorrespondenceDistance(0.1);
        gicp.setEuclideanFitnessEpsilon(5e-5);
        gicp.setRANSACOutlierRejectionThreshold(0.05);
        
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
        
        // TODO: use the features to find the best match for the segments and choose the best one.
        
        // Check if cloud is empty
        if (preprocessed_cloud->size() < 100) {
            publishEmptyPose(cloud_msg);
            return;
        }
        
        // Visualize preprocessed cloud if debugging enabled
        publishDebugCloud(preprocessed_cloud, cloud_msg);
        
        // Perform registration
        Eigen::Matrix4f final_transformation = performRegistration(preprocessed_cloud, cloud_msg);
        
        // Publish and broadcast results
        publishRegistrationResults(final_transformation, cloud_msg);

        // publish the filtered cloud
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*preprocessed_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = cloud_msg->header;
        filtered_cloud_msg.header.frame_id = cloud_msg->header.frame_id;
        cloud_debug_pub_->publish(filtered_cloud_msg);
        
        
        auto end_time = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), 
            "Total processing time: %.3f s",
            std::chrono::duration<double>(end_time - start_time).count());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
        tracking_initialized_ = false;  // Reset tracking on error
    }
}

// Publish empty pose when cloud is insufficient
void publishEmptyPose(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    RCLCPP_INFO(this->get_logger(), "Scene point cloud is empty, skipping registration");

    empty_pose_.header.stamp = cloud_msg->header.stamp;
    empty_pose_.header.frame_id = "map";
    empty_pose_.pose = ros_utils::matrix_to_pose(Eigen::Matrix4f::Identity());
    icp_result_publisher_->publish(empty_pose_);
    goicp_result_publisher_->publish(empty_pose_);
    tracking_initialized_ = false;
}



// Publish debug point cloud for visualization
void publishDebugCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    if (save_debug_clouds_) {
        sensor_msgs::msg::PointCloud2 debug_cloud_msg;
        pcl::toROSMsg(*cloud, debug_cloud_msg);
        debug_cloud_msg.header = cloud_msg->header;
        cloud_debug_pub_->publish(debug_cloud_msg);
    }
}

// Perform registration (Go-ICP, GICP, or both)
Eigen::Matrix4f performRegistration(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& preprocessed_cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    Eigen::Matrix4f final_transformation;
    float gicp_score = 0.0f;
    
    // Determine if we need to use Go-ICP
    bool run_goicp = use_goicp_ && (!tracking_initialized_ || previous_gicp_score_ > gicp_fitness_threshold_);
    
    // First try GICP with previous transform if we have good tracking
    if (!run_goicp && tracking_initialized_) {
        RCLCPP_INFO(this->get_logger(), "Using GICP with previous transformation");
        
        bool gen_icp_converged = runGICP(
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
    if (run_goicp) {
        final_transformation = runGoICP(preprocessed_cloud, cloud_msg);
    }
    
    return final_transformation;
}

// Run Go-ICP registration and GICP refinement
Eigen::Matrix4f runGoICP(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& preprocessed_cloud,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    RCLCPP_INFO(this->get_logger(), "Using Go-ICP for global alignment");
    
    // Run Go-ICP registration
    Eigen::Matrix4f alignment_transform = goicp_wrapper_->registerPointClouds(
        model_cloud_,
        preprocessed_cloud,
        2000,  // Max target points 
        goicp_debug_
    );
    
    // Get Go-ICP statistics
    float goicp_error = goicp_wrapper_->getLastError();
    float goicp_time = goicp_wrapper_->getLastRegistrationTime();
    
    RCLCPP_INFO(this->get_logger(), 
        "Go-ICP completed in %.3f seconds with error: %.5f", 
        goicp_time, goicp_error);
    
    // Publish Go-ICP result
    publishGoICPResult(alignment_transform, cloud_msg);
    
    // Now refine with GICP using Go-ICP result as initial guess
    Eigen::Matrix4f final_transformation;
    float gicp_score = 0.0f;
    
    bool gen_icp_converged = runGICP(
        model_cloud_, 
        preprocessed_cloud, 
        alignment_transform, 
        final_transformation, 
        gicp_score
    );
    
    if (gen_icp_converged) {
        RCLCPP_INFO(this->get_logger(), "GICP refinement converged with score: %f", gicp_score);
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

// Publish Go-ICP registration result
void publishGoICPResult(
    const Eigen::Matrix4f& alignment_transform,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    geometry_msgs::msg::PoseStamped goicp_pose;
    goicp_pose.header.stamp = cloud_msg->header.stamp;
    goicp_pose.header.frame_id = cloud_msg->header.frame_id;
    goicp_pose.pose = ros_utils::matrix_to_pose(alignment_transform);
    goicp_result_publisher_->publish(goicp_pose);
    
    // Broadcast the Go-ICP transformation for visualization
    ros_utils::broadcast_transform(
        tf_broadcaster_,
        alignment_transform,
        cloud_msg->header.stamp,
        cloud_msg->header.frame_id,
        object_frame_ + "_goicp"
    );
}

// Publish final registration results
void publishRegistrationResults(
    const Eigen::Matrix4f& final_transformation,
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    // Publish the final transformation
    geometry_msgs::msg::PoseStamped aligned_pose;
    aligned_pose.header.stamp = cloud_msg->header.stamp;
    aligned_pose.header.frame_id = cloud_msg->header.frame_id;
    aligned_pose.pose = ros_utils::matrix_to_pose(final_transformation);
    
    icp_result_publisher_->publish(aligned_pose);
    
    // Broadcast the final transformation
    ros_utils::broadcast_transform(
        tf_broadcaster_,
        final_transformation,
        cloud_msg->header.stamp,
        cloud_msg->header.frame_id,
        object_frame_
    );
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
        pass_z.setInputCloud(input_cloud);
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
        auto clustering_result = pcl_utils::cluster_point_cloud(segmentation_result.remaining_cloud, this->get_logger(), cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        
        
        // TODO: apply FPFH features to the segments and model pcd. Use the clustering_result.individual_clusters to find the normals and 3D descriptor, then compare with model descriptor.


        // If you want to publish the planes cloud:
        if (save_debug_clouds_ && segmentation_result.planes_cloud->size() > 0) {
            sensor_msgs::msg::PointCloud2 pointcloud_msg;
            pcl::toROSMsg(*segmentation_result.remaining_cloud, pointcloud_msg);
            pointcloud_msg.header = cloud_msg->header;
            plane_debug_pub_->publish(pointcloud_msg);

            pcl::toROSMsg(*segmentation_result.planes_cloud, pointcloud_msg);
            pointcloud_msg.header = cloud_msg->header;
            filtered_plane_debug_pub_->publish(pointcloud_msg);

            pcl::toROSMsg(*clustering_result.colored_cloud, pointcloud_msg);
            pointcloud_msg.header = cloud_msg->header;
            clustered_plane_debug_pub_->publish(pointcloud_msg);


        }
        
        
        return filtered_cloud;
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
    
    // Callback groups for concurrent execution
    rclcpp::CallbackGroup::SharedPtr callback_group_subscription_;
    rclcpp::CallbackGroup::SharedPtr callback_group_processing_;
    
    // Parameters
    double voxel_size_;
    bool save_debug_clouds_;
    std::string object_frame_;
    int processing_period_ms_;
    bool use_goicp_;
    bool goicp_debug_;
    double gicp_fitness_threshold_;
    
    // New clustering parameters
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;

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
    
    // Pose stamped
    geometry_msgs::msg::PoseStamped empty_pose_;

    
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