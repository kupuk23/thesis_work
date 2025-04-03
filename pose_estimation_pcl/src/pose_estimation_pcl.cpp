#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <chrono>
#include <cfloat>  // For FLT_MAX

// include tf2 buffer for lookup transform
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include "pose_estimation_pcl/pcl_utils.hpp"  // Our utility header

using namespace std::chrono_literals;

class PoseEstimationPCL : public rclcpp::Node {
public:
    PoseEstimationPCL() : Node("pose_estimation_pcl") {
        // Initialize parameters
        voxel_size_ = this->declare_parameter<double>("voxel_size", 0.01);  // Default voxel size
        save_debug_clouds_ = this->declare_parameter<bool>("save_debug_clouds", false);
        debug_path_ = this->declare_parameter<std::string>("debug_path", "/home/tafarrel/debugPCD_cpp");
        object_frame_ = this->declare_parameter<std::string>("object_frame", "grapple");
        processing_period_ms_ = 100;  // 10 Hz default
    
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
        
        // Load model cloud
        if (object_frame_ == "grapple") 
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/grapple_fixture_down.pcd", this->get_logger());
        else
            model_cloud_ = pcl_utils::loadModelPCD("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", this->get_logger());

        // Initialize processing timer in separate callback group
        processing_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(processing_period_ms_),
            std::bind(&PoseEstimationPCL::process_data, this),
            callback_group_processing_);
            
        // Initialize empty pose
        empty_pose_.header.frame_id = "camera_depth_optical_frame";  // Default frame
        
        // Initialize flags and mutex
        has_new_data_ = false;
        
        RCLCPP_INFO(this->get_logger(), "PoseEstimationPCL node initialized with separated callback groups");
    }

private:
    // Callback for receiving point cloud data (lightweight)
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg) {
        try {
            // Store the pointcloud and perform lightweight operations
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            latest_cloud_msg_ = pointcloud_msg;
            
            // Get the initial transformation from the object pose using TF2 lookup
            latest_initial_transform_ = pcl_utils::transform_obj_pose(
                pointcloud_msg, 
                *tf_buffer_, 
                object_frame_,
                this->get_logger()
            );

            // TODO: add gaussian noise to the initial transform
            
            // Broadcast the transformation for visualization
            pcl_utils::broadcast_transform(
                tf_broadcaster_,
                latest_initial_transform_,
                pointcloud_msg->header.stamp,
                pointcloud_msg->header.frame_id,
                object_frame_ + "_initial"
            );
            
            
            // Update flag to indicate new data is available
            has_new_data_ = true;
            
            RCLCPP_DEBUG(this->get_logger(), "Received new point cloud data");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in pointcloud callback: %s", e.what());
        }
    }
    
    // Separate timer callback for processing data (heavy processing)
    void process_data() {
        // Check if we have new data to process
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
        Eigen::Matrix4f initial_transform;
        
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (!has_new_data_ || !latest_cloud_msg_) {
                return;  // No new data to process
            }
            
            // Copy the data we need for processing
            cloud_msg = latest_cloud_msg_;
            initial_transform = latest_initial_transform_;
            has_new_data_ = false;  // Reset flag
        }
        
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Convert point cloud and process
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pcl_utils::convertPointCloud2ToPCL(cloud_msg);
            
            auto conversion_time = std::chrono::high_resolution_clock::now();
            // RCLCPP_INFO(this->get_logger(), 
            //     "Pointcloud converted for processing (%.3f s)",
            //     std::chrono::duration<double>(conversion_time - start_time).count());
            
            // Save the cloud for debugging if requested
            if (save_debug_clouds_) {
                std::string original_filename = debug_path_ + "/original_cloud.pcd";
                // pcl_utils::saveToPCD(cloud, original_filename, this->get_logger());
                
                
            }
            
            // Preprocess the point cloud
            auto preprocessed_cloud = preprocess_pointcloud(cloud);
            
            // auto processing_time = std::chrono::high_resolution_clock::now();
            // RCLCPP_INFO(this->get_logger(), 
            //     "Pointcloud processed --> %ld points (%.3f s)",
            //     preprocessed_cloud->size(),
            //     std::chrono::duration<double>(processing_time - conversion_time).count());
            
            // Check if cloud is empty
            if (preprocessed_cloud->size() < 100) {
                RCLCPP_INFO(this->get_logger(), "Scene point cloud is empty, skipping ICP");
                return;
            }
            
            // Save processed cloud for debugging if requested
            if (save_debug_clouds_) {
                std::string processed_filename = debug_path_ + "/processed_cloud.pcd";
                // Also republish the cloud for RViz visualization
                sensor_msgs::msg::PointCloud2 debug_cloud_msg;
                pcl::toROSMsg(*preprocessed_cloud, debug_cloud_msg);
                debug_cloud_msg.header = cloud_msg->header;
                cloud_debug_pub_->publish(debug_cloud_msg);
            }
            
            // Check if we got a valid transformation
            if (initial_transform.isIdentity()) {
                RCLCPP_WARN(this->get_logger(), "Using identity as initial transformation");
            }

            // Perform ICP alignment
            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
            
            icp.setInputSource(model_cloud_);
            icp.setInputTarget(preprocessed_cloud);
            icp.setUseReciprocalCorrespondences(true);
            // Set ICP parameters
            icp.setMaximumIterations(50);
            icp.setTransformationEpsilon(1e-5);
            icp.setMaxCorrespondenceDistance(0.1);
            icp.setEuclideanFitnessEpsilon(5e-7);
            icp.setRANSACOutlierRejectionThreshold(0.05);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            icp.align(*aligned_cloud, initial_transform);

            // Check if ICP converged
            if (icp.hasConverged()) {
                RCLCPP_INFO(this->get_logger(), "ICP converged with score: %f", icp.getFitnessScore());
                
                // Get the final transformation
                Eigen::Matrix4f final_transformation = icp.getFinalTransformation();
                
                // Convert to pose and publish
                geometry_msgs::msg::PoseStamped aligned_pose;
                aligned_pose.header.stamp = cloud_msg->header.stamp;
                aligned_pose.header.frame_id = cloud_msg->header.frame_id;
                aligned_pose.pose = pcl_utils::matrix_to_pose(final_transformation);
                
                icp_result_publisher_->publish(aligned_pose);
                
                // Optionally save the aligned cloud for debugging
                if (save_debug_clouds_) {
                    std::string aligned_filename = debug_path_ + "/aligned_cloud.pcd";
                    // pcl_utils::saveToPCD(aligned_cloud, aligned_filename, this->get_logger());
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "ICP did not converge");
                // Publish the initial pose as fallback
                geometry_msgs::msg::PoseStamped initial_pose;
                initial_pose.header.stamp = cloud_msg->header.stamp;
                initial_pose.header.frame_id = cloud_msg->header.frame_id;
                initial_pose.pose = pcl_utils::matrix_to_pose(initial_transform);
                icp_result_publisher_->publish(initial_pose);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            // RCLCPP_INFO(this->get_logger(), 
            //     "Total processing time: %.3f s",
            //     std::chrono::duration<double>(end_time - start_time).count());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
        }
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr preprocess_pointcloud(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud) {
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        
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
        
        // Downsample using voxel grid
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setInputCloud(filtered_cloud);
        voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        voxel_grid.filter(*filtered_cloud);
        
        return filtered_cloud;
    }

    // Node members
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr icp_result_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_debug_pub_;
    rclcpp::TimerBase::SharedPtr processing_timer_;
    
    // Callback groups for concurrent execution
    rclcpp::CallbackGroup::SharedPtr callback_group_subscription_;
    rclcpp::CallbackGroup::SharedPtr callback_group_processing_;
    
    // Parameters
    double voxel_size_;
    bool save_debug_clouds_;
    std::string debug_path_;
    std::string object_frame_;
    int processing_period_ms_;
    
    // TF2 buffer and listener
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_;
    
    // Data storage with thread synchronization
    std::mutex data_mutex_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_msg_;
    Eigen::Matrix4f latest_initial_transform_;
    bool has_new_data_;
    
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