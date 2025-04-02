#include "pose_estimation_pcl/pcl_utils.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace pcl_utils {

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

geometry_msgs::msg::Pose matrix_to_pose(const Eigen::Matrix4f& transform) {
    // Manual conversion from Eigen matrix to ROS pose
    Eigen::Quaternionf q(transform.block<3,3>(0,0));
    
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = transform(0,3);
    pose_msg.position.y = transform(1,3);
    pose_msg.position.z = transform(2,3);
    pose_msg.orientation.x = q.x();
    pose_msg.orientation.y = q.y();
    pose_msg.orientation.z = q.z();
    pose_msg.orientation.w = q.w();
    
    return pose_msg;
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

} // namespace pcl_utils