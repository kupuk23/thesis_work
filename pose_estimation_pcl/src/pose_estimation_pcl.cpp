#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <cfloat>  // For FLT_MAX

#include "pose_estimation_pcl/pcl_utils.hpp"  // Our utility header

using namespace std::chrono_literals;

class PoseEstimationPCL : public rclcpp::Node {
public:
    PoseEstimationPCL() : Node("pose_estimation_pcl") {
        // Initialize parameters
        voxel_size_ = this->declare_parameter<double>("voxel_size", 0.01);  // Default voxel size
        save_debug_clouds_ = this->declare_parameter<bool>("save_debug_clouds", true);
        debug_path_ = this->declare_parameter<std::string>("debug_path", "/home/tafarrel/debugPCD_cpp");
        
        // Initialize publishers and subscribers
        pointcloud_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/points", 10, 
            std::bind(&PoseEstimationPCL::pointcloud_callback, this, std::placeholders::_1));
        
        icp_result_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose/icp_result", 10);
        
        // Create a publisher for debugged point clouds if needed
        cloud_debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/debug_pointcloud", 1);
        
        // TODO: Load model PCD file
        // model_cloud_ = pcl_utils::loadModelPCD("path/to/model.pcd", this->get_logger());
            
        RCLCPP_INFO(this->get_logger(), "PoseEstimationPCL node initialized");
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Use our custom function to convert PointCloud2 to PCL
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pcl_utils::convertPointCloud2ToPCL(pointcloud_msg);
            
            auto conversion_time = std::chrono::high_resolution_clock::now();
            RCLCPP_INFO(this->get_logger(), 
                "Pointcloud retrieved from camera.. (%.3f s)",
                std::chrono::duration<double>(conversion_time - start_time).count());
            
            // Save the cloud for debugging if requested
            if (save_debug_clouds_) {
                std::string original_filename = debug_path_ + "/original_cloud.pcd";
                // pcl_utils::saveToPCD(cloud, original_filename, this->get_logger());
                
                // Also republish the cloud for RViz visualization
                sensor_msgs::msg::PointCloud2 debug_cloud_msg;
                pcl::toROSMsg(*cloud, debug_cloud_msg);
                debug_cloud_msg.header = pointcloud_msg->header;
                cloud_debug_pub_->publish(debug_cloud_msg);
            }
            
            // Preprocess the point cloud
            auto preprocessed_cloud = preprocess_pointcloud(cloud);
            
            auto processing_time = std::chrono::high_resolution_clock::now();
            RCLCPP_INFO(this->get_logger(), 
                "Pointcloud processed --> %ld points (%.3f s)",
                preprocessed_cloud->size(),
                std::chrono::duration<double>(processing_time - conversion_time).count());
            
            // Check if cloud is empty
            if (preprocessed_cloud->size() < 100) {
                RCLCPP_INFO(this->get_logger(), "Scene point cloud is empty");
                return;
            }
            
            // Save processed cloud for debugging if requested
            if (save_debug_clouds_) {
                std::string processed_filename = debug_path_ + "/processed_cloud.pcd";
                // pcl_utils::saveToPCD(preprocessed_cloud, processed_filename, this->get_logger());
            }
            
            // Here you would implement the ICP process
            
            // Create and publish an empty pose message
            geometry_msgs::msg::PoseStamped empty_pose;
            empty_pose.header.stamp = pointcloud_msg->header.stamp;
            empty_pose.header.frame_id = "map";
            empty_pose.pose.orientation.w = 1.0;  // Identity orientation
            
            icp_result_publisher_->publish(empty_pose);
            
            RCLCPP_INFO(this->get_logger(), "Published empty pose (ICP not implemented yet)");
            
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
    
    // Parameters
    double voxel_size_;
    bool save_debug_clouds_;
    std::string debug_path_;
    
    // Point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseEstimationPCL>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}