#include "pose_estimation_pcl/pcl_utils.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.hpp>

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

Eigen::Matrix4f pose_to_matrix(const geometry_msgs::msg::Pose& pose) {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    
    // Convert position
    mat(0, 3) = pose.position.x;
    mat(1, 3) = pose.position.y;
    mat(2, 3) = pose.position.z;
    
    // Convert orientation (quaternion to rotation matrix)
    Eigen::Quaternionf q(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z
    );
    mat.block<3, 3>(0, 0) = q.toRotationMatrix();
    
    return mat;
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



Eigen::Matrix4f transform_obj_pose(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc2_msg,
    tf2_ros::Buffer& tf_buffer,
    const std::string& obj_frame,
    const rclcpp::Logger& logger) {
    
    try {
        // Check if transforms are available
        if (!tf_buffer.canTransform(pc2_msg->header.frame_id, "map", tf2::TimePointZero) || 
            !tf_buffer.canTransform("map", obj_frame, tf2::TimePointZero)) {
            RCLCPP_WARN(logger, "Cannot transform between required frames");
            return Eigen::Matrix4f::Identity();
        }
        
        // Get transform from map to camera frame
        geometry_msgs::msg::TransformStamped map_T_cam = tf_buffer.lookupTransform(
            pc2_msg->header.frame_id,  // target frame
            "map",                     // source frame
            tf2::TimePointZero,        // time
            tf2::durationFromSec(1.0)  // timeout
        );
        
        // Get transform from object frame to map
        geometry_msgs::msg::TransformStamped obj_T_map = tf_buffer.lookupTransform(
            "map",                     // target frame
            obj_frame,                 // source frame
            tf2::TimePointZero,        // time
            tf2::durationFromSec(1.0)  // timeout
        );
        
        // Convert transform to pose
        geometry_msgs::msg::PoseStamped obj_pose_map;
        obj_pose_map.header.stamp = pc2_msg->header.stamp;
        obj_pose_map.header.frame_id = "map";
        obj_pose_map.pose.position.x = obj_T_map.transform.translation.x;
        obj_pose_map.pose.position.y = obj_T_map.transform.translation.y;
        obj_pose_map.pose.position.z = obj_T_map.transform.translation.z;
        obj_pose_map.pose.orientation = obj_T_map.transform.rotation;
        
        // Transform the pose to camera frame
        geometry_msgs::msg::PoseStamped obj_pose_camera;
        tf2::doTransform(obj_pose_map, obj_pose_camera, map_T_cam);
        
        // Create result message (not returned but similar to Python function)
        geometry_msgs::msg::PoseStamped result_msg;
        result_msg.header.stamp = pc2_msg->header.stamp;
        result_msg.header.frame_id = pc2_msg->header.frame_id;
        result_msg.pose = obj_pose_camera.pose;
        
        // Convert pose to matrix
        Eigen::Matrix4f handrail_pose_matrix = pose_to_matrix(obj_pose_camera.pose);
        return handrail_pose_matrix;
        
    } catch (const tf2::TransformException& e) {
        RCLCPP_ERROR(logger, "Error transforming object pose: %s", e.what());
        return Eigen::Matrix4f::Identity();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "General error transforming object pose: %s", e.what());
        return Eigen::Matrix4f::Identity();
    }
}

geometry_msgs::msg::Pose matrixToPose(const Eigen::Matrix4d& matrix) {
    geometry_msgs::msg::Pose pose;
    
    // Extract translation
    pose.position.x = matrix(0, 3);
    pose.position.y = matrix(1, 3);
    pose.position.z = matrix(2, 3);
    
    // Extract rotation matrix
    Eigen::Matrix3d rotation_matrix = matrix.block<3, 3>(0, 0);
    
    // Convert rotation matrix to quaternion
    tf2::Matrix3x3 tf_rotation_matrix(
        rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2),
        rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2),
        rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2)
    );
    
    tf2::Quaternion quaternion;
    tf_rotation_matrix.getRotation(quaternion);
    
    // Set quaternion in pose
    pose.orientation.x = quaternion.x();
    pose.orientation.y = quaternion.y();
    pose.orientation.z = quaternion.z();
    pose.orientation.w = quaternion.w();
    
    return pose;
}

} // namespace pcl_utils