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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <chrono>
#include <cfloat>  // For FLT_MAX
#include <thread> 

// include tf2 buffer for lookup transform
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include "pose_estimation_pcl/pcl_utils.hpp"    // Our utility header
#include "pose_estimation_pcl/go_icp_wrapper.hpp"  // Our Go-ICP wrapper

using namespace std::chrono_literals;

struct PlaneInfo {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    size_t num_points;
    pcl::ModelCoefficients::Ptr coefficients;
    int index;  // To track the original detection order
};

static const std::vector<std::array<uint8_t, 3>> DEFAULT_COLORS = {
    {255, 0, 0},   // Red
    {0, 255, 0},   // Green
    {0, 0, 255},   // Blue
    {255, 255, 0}, // Yellow
    {255, 0, 255}, // Magenta
    {0, 255, 255}  // Cyan
};

class PoseEstimationPCL : public rclcpp::Node {
public:
    PoseEstimationPCL() : Node("pose_estimation_pcl") {
        // Initialize parameters
        voxel_size_ = this->declare_parameter<double>("voxel_size", 0.05);  // Default voxel size
        save_debug_clouds_ = this->declare_parameter<bool>("save_debug_clouds", false);
        debug_path_ = this->declare_parameter<std::string>("debug_path", "/home/tafarrel/debugPCD_cpp");
        object_frame_ = this->declare_parameter<std::string>("object_frame", "grapple");
        processing_period_ms_ = 100;  // 10 Hz default
        use_goicp_ = this->declare_parameter<bool>("use_goicp", false);
        goicp_debug_ = this->declare_parameter<bool>("goicp_debug", false);
        
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

     // Struct to hold the plane cloud
    struct PlaneSegmentationResult {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_plane_cloud;
    };

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
        
        // TODO: add segmentation to the preprocessed cloud
        // TODO: apply FPFH features to the segments and model pcd.
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
    empty_pose_.pose = pcl_utils::matrix_to_pose(Eigen::Matrix4f::Identity());
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
    goicp_pose.pose = pcl_utils::matrix_to_pose(alignment_transform);
    goicp_result_publisher_->publish(goicp_pose);
    
    // Broadcast the Go-ICP transformation for visualization
    pcl_utils::broadcast_transform(
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
    aligned_pose.pose = pcl_utils::matrix_to_pose(final_transformation);
    
    icp_result_publisher_->publish(aligned_pose);
    
    // Broadcast the final transformation
    pcl_utils::broadcast_transform(
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

        PlaneSegmentationResult segmentation_result = detect_and_remove_planes(filtered_cloud, this->get_logger(), true);
        
        // If you want to publish the planes cloud:
        if (save_debug_clouds_ && segmentation_result.planes_cloud->size() > 0) {
            sensor_msgs::msg::PointCloud2 planes_msg;
            pcl::toROSMsg(*segmentation_result.remaining_cloud, planes_msg);
            planes_msg.header = cloud_msg->header;
            plane_debug_pub_->publish(planes_msg);

            pcl::toROSMsg(*segmentation_result.planes_cloud, planes_msg);
            planes_msg.header = cloud_msg->header;
            filtered_plane_debug_pub_->publish(planes_msg);


        }
        
        
        return filtered_cloud;
    }

   
    PlaneSegmentationResult detect_and_remove_planes(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud,
        const rclcpp::Logger& logger,
        bool colorize_planes = true) {
        
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
        int remaining_points = working_cloud->size();
        const int min_plane_points = 800;  // Minimum points to consider a plane
        const float min_remaining_percent = 0.2;  // Stop if only 40% of points remain
        const int max_planes = 3;  // Maximum number of planes to extract
        
        

        
        RCLCPP_INFO(logger, "Starting plane detection on cloud with %d points", remaining_points);
        
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
                
                RCLCPP_INFO(logger, "Coloring plane %d with RGB(%d,%d,%d)", plane_count, r, g, b);
                
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
            
            RCLCPP_INFO(logger, "Plane %d: extracted %lu points (%.1f%% of original)",
                       plane_count, inliers->indices.size(),
                       100.0f * inliers->indices.size() / input_cloud->size());
            
            // Print plane equation: ax + by + cz + d = 0
            RCLCPP_INFO(logger, "Plane equation: %.2fx + %.2fy + %.2fz + %.2f = 0",
                       coefficients->values[0], coefficients->values[1],
                       coefficients->values[2], coefficients->values[3]);
            
            // Update the working cloud for next iteration
            working_cloud = remaining_cloud;
        }
        
        // Now that we've found all planes and identified the largest one,
        // remove only the largest plane from the input cloud
        if (largest_plane_size > 0) {
            *result.largest_plane_cloud = *largest_plane;
            *result.remaining_cloud = *with_largest_plane_removed;
            RCLCPP_INFO(logger, "Removed largest plane with %lu points. Remaining cloud has %lu points.",
                       largest_plane_size, result.remaining_cloud->size());
        } else {
            // No planes found, return the original cloud
            *result.remaining_cloud = *input_cloud;
            RCLCPP_INFO(logger, "No planes found to remove");
        }
        
        RCLCPP_INFO(logger, "Plane detection complete. Found %d planes, visualizing %lu points in planes cloud.",
                   plane_count, result.planes_cloud->size());
        
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
    
    // Callback groups for concurrent execution
    rclcpp::CallbackGroup::SharedPtr callback_group_subscription_;
    rclcpp::CallbackGroup::SharedPtr callback_group_processing_;
    
    // Parameters
    double voxel_size_;
    bool save_debug_clouds_;
    std::string debug_path_;
    std::string object_frame_;
    int processing_period_ms_;
    bool use_goicp_;
    bool goicp_debug_;
    double gicp_fitness_threshold_;
    
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