#include <pose_estimation_pcl/ros_utils.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <random>
#include <string>

namespace ros_utils {
void broadcast_transform(
    std::shared_ptr<tf2_ros::TransformBroadcaster>& broadcaster,
    const Eigen::Matrix4f& transform,
    const rclcpp::Time& header_stamp,
    const std::string& header_frame_id,
    const std::string& child_frame_id) {
    
    geometry_msgs::msg::TransformStamped transform_stamped = 
        create_transform_stamped(transform, header_stamp, header_frame_id, child_frame_id);
    
    broadcaster->sendTransform(transform_stamped);
}



geometry_msgs::msg::TransformStamped create_transform_stamped(
    const Eigen::Matrix4f& transform,
    const rclcpp::Time& header_stamp,
    const std::string& header_frame_id,
    const std::string& child_frame_id) {
    
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = header_stamp;
    transform_stamped.header.frame_id = header_frame_id;
    transform_stamped.child_frame_id = child_frame_id;
    
    // Set translation
    transform_stamped.transform.translation.x = transform(0, 3);
    transform_stamped.transform.translation.y = transform(1, 3);
    transform_stamped.transform.translation.z = transform(2, 3);
    
    // Set rotation (convert matrix to quaternion)
    Eigen::Quaternionf q(transform.block<3, 3>(0, 0));
    transform_stamped.transform.rotation.x = q.x();
    transform_stamped.transform.rotation.y = q.y();
    transform_stamped.transform.rotation.z = q.z();
    transform_stamped.transform.rotation.w = q.w();
    
    return transform_stamped;
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


Eigen::Matrix4f transform_obj_pose(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pc2_msg,
    tf2_ros::Buffer& tf_buffer,
    const std::string& obj_frame,
    const rclcpp::Logger& logger) {
    
    try {
        // Check if transforms are available
        if (!tf_buffer.canTransform(pc2_msg->header.frame_id, obj_frame, tf2::TimePointZero)) {
            RCLCPP_WARN(logger, "Cannot transform between required frames");
            return Eigen::Matrix4f::Identity();
        }
        
        // Get transform from map to camera frame
        geometry_msgs::msg::TransformStamped obj_T_cam = tf_buffer.lookupTransform(
            pc2_msg->header.frame_id,  // target frame
            obj_frame,                     // source frame
            tf2::TimePointZero,        // time
            tf2::durationFromSec(1.0)  // timeout
        );
        
        // Transform the object to the camera frame transformStamped into Pose
        geometry_msgs::msg::Pose obj_pose_camera;
        obj_pose_camera.orientation = obj_T_cam.transform.rotation;
        obj_pose_camera.position.x = obj_T_cam.transform.translation.x;
        obj_pose_camera.position.y = obj_T_cam.transform.translation.y;
        obj_pose_camera.position.z = obj_T_cam.transform.translation.z;

        
        // Create result message (not returned but similar to Python function)
        geometry_msgs::msg::PoseStamped result_msg;
        result_msg.header.stamp = pc2_msg->header.stamp;
        result_msg.header.frame_id = pc2_msg->header.frame_id;
        result_msg.pose = obj_pose_camera;
        
        // Convert pose to matrix
        Eigen::Matrix4f handrail_pose_matrix = pose_to_matrix(obj_pose_camera);
        return handrail_pose_matrix;
        
    } catch (const tf2::TransformException& e) {
        RCLCPP_ERROR(logger, "Error transforming object pose: %s", e.what());
        return Eigen::Matrix4f::Identity();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "General error transforming object pose: %s", e.what());
        return Eigen::Matrix4f::Identity();
    }
}


Eigen::Matrix4f apply_noise_to_transform(
    const Eigen::Matrix4f& transform, 
    float t_std, 
    float r_std) {
    
    // Check if transform is valid
    if (transform.isIdentity()) {
        return transform;
    }
    
    // Create a copy of the transformation
    Eigen::Matrix4f noisy_transform = transform;
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> t_dist(0.0, t_std); // Translation noise distribution
    std::normal_distribution<float> r_dist(0.0, r_std); // Rotation noise distribution
    
    // Add noise to translation
    for (int i = 0; i < 3; ++i) {
        noisy_transform(i, 3) += t_dist(gen);
    }
    
    // Extract current rotation matrix
    Eigen::Matrix3f current_rot = noisy_transform.block<3, 3>(0, 0);
    
    // Convert to Euler angles (XYZ order)
    Eigen::Vector3f euler = current_rot.eulerAngles(0, 1, 2); // x, y, z
    
    // Add noise to Euler angles
    euler[0] += r_dist(gen);
    euler[1] += r_dist(gen);
    euler[2] += r_dist(gen);
    
    // Convert back to rotation matrix
    Eigen::Matrix3f noisy_rot;
    noisy_rot = Eigen::AngleAxisf(euler[0], Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(euler[1], Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(euler[2], Eigen::Vector3f::UnitZ());
    
    // Update rotation part of transformation matrix
    noisy_transform.block<3, 3>(0, 0) = noisy_rot;
    
    return noisy_transform;
}

} // namespace ros_utils