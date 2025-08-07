#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener

import tf_transformations as tft



#calculate the error between pose from mocap with ICP (only in 2D, x and y coordinates)

class PoseErrorCalculator(Node):
    def __init__(self):
        super().__init__('pose_error_calc_node')
        self.get_logger().info("Pose Error Calculator Node has been started.")
        self.mocap_pose = None
        self.goicp_pose = None
        self.icp_pose = None

        self.icp_sub = self.create_subscription(
            PoseStamped, "/pose/icp_result", self.icp_pose_callback, 10
        )

        self.goicp_sub = self.create_subscription(
            PoseStamped, "/pose/goicp_result", self.goicp_pose_callback, 10
        )

        self.mocap_sub = self.create_subscription(
            PoseStamped, "/docking_st/pose", self.mocap_cb, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.5, self.timer_callback)
        

    def timer_callback(self):
        if self.icp_pose is None or self.goicp_pose is None:
            self.get_logger().warn("Waiting for poses from ICP and GoICP.")
            return

        frame_id = "zed_camera_link"  # Replace with the actual frame ID if needed
        transform = lookup_transform(
            self.tf_buffer, "map", frame_id
        )
        if transform is None:
            return None, None

        goicp_pose_on_map = do_transform_pose(
            self.goicp_pose, transform)
        icp_pose_on_map = do_transform_pose(
            self.icp_pose, transform)
        # Calculate the error between the two poses

        init_lin_error, init_angle_error = self.calculate_pose_error(self.mocap_pose, goicp_pose_on_map)
        refined_lin_error, refined_angle_error = self.calculate_pose_error(self.mocap_pose, icp_pose_on_map)


        # Log the errors
        self.get_logger().info(f"Initial Linear Error: {init_lin_error:.4f}, Angular Error: {init_angle_error:.4f} | Refined Linear Error: {refined_lin_error:.4f}, Refined Angular Error: {refined_angle_error:.4f}")
    def icp_pose_callback(self, msg):
        self.icp_pose = msg.pose

    def mocap_cb(self, msg):
        self.mocap_pose = msg.pose

    def goicp_pose_callback(self, msg):
        self.goicp_pose = msg.pose

    def quaternion_angular_distance(self, q1_ros, q2_ros):
        """
        Calculate angular distance between two quaternions in degrees
        """

        q1 = np.array([q1_ros.x, q1_ros.y, q1_ros.z, q1_ros.w])
        q2 = np.array([q2_ros.x, q2_ros.y, q2_ros.z, q2_ros.w])

        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Handle quaternion double cover (q and -q represent same rotation)
        dot_product = abs(np.dot(q1, q2))
        dot_product = np.clip(dot_product, 0.0, 1.0)  # Handle numerical errors
        
        angular_error = 2 * np.arccos(dot_product)  # in radians
        angular_error_degrees = np.degrees(angular_error)  # convert to degrees
        return angular_error_degrees

    def calculate_pose_error(self, pose_mocap, pose_icp):
        """
        Calculate the error between the mocap pose and the ICP pose.
        
        :param pose_mocap: Pose from mocap system
        :param pose_icp: Pose from ICP algorithm
        :return: linear error,  angular error
        """
        lin_error = np.linalg.norm(
            np.array([pose_mocap.position.x - pose_icp.position.x,
                      pose_mocap.position.y - pose_icp.position.y])
        )
        # convert quaternions to angles
        angle_error = self.quaternion_angular_distance(pose_mocap.orientation, pose_icp.orientation)
        return lin_error, angle_error
    


def main(args=None):
    rclpy.init(args=args)

    pose_error_calculator = PoseErrorCalculator()

    try:
        rclpy.spin(pose_error_calculator)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        pose_error_calculator.destroy_node()
        rclpy.shutdown()


    
def lookup_transform(tf_buffer, target_frame, source_frame):
    """
    Lookup the transform between two frames using the provided tf_buffer.
    Parameters:
    ----------
    tf_buffer : tf2_ros.Buffer
        The TF buffer to use for looking up transforms
    target_frame : str
        The target frame to transform to
    source_frame : str
        The source frame to transform from
    Returns:
    --------
    TransformStamped
        The transform between the two frames, or None if not found
    """
    # Check if the transform is available    
    try:
        if not tf_buffer.can_transform(
                        target_frame, source_frame, rclpy.time.Time()
                    ):
            transformed_pose_stamped = None
        else:
            transform = tf_buffer.lookup_transform(
                target_frame,  # target frame
                source_frame,  # source frame
                rclpy.time.Time(),  # get the latest transform
                rclpy.duration.Duration(seconds=1.0),  # timeout
            )
            return transform
    except Exception as e:
        print(f"Transform error: {e}")
        return None


if __name__ == "__main__":
    main()

