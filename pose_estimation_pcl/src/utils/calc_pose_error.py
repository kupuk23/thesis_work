#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np



#calculate the error between pose from mocap with ICP (only in 2D, x and y coordinates)

class PoseErrorCalculator(Node):
    def __init__(self):
        super().__init__('pose_error_calculator_node')
        self.get_logger().info("Pose Error Calculator Node has been started.")

    def calculate_error(self, pose_mocap, pose_icp):
        """
        Calculate the error between the mocap pose and the ICP pose.
        
        :param pose_mocap: Pose from mocap system
        :param pose_icp: Pose from ICP algorithm
        :return: Error value
        """
        lin_error = np.linalg.norm(
            np.array([pose_mocap.position.x - pose_icp.position.x,
                      pose_mocap.position.y - pose_icp.position.y])
        )
        # convert quaternions to angles
        angle_mocap = np.arctan2(2 * (pose_mocap.orientation.z * pose_mocap.orientation.w + pose_mocap.orientation.x * pose_mocap.orientation.y),
                                 1 - 2 * (pose_mocap.orientation.y ** 2 + pose_mocap.orientation.z ** 2))
        
    
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


if __name__ == "__main__":
    main()

