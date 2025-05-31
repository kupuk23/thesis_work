#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.linalg import pinv
from ibvs_testing.detect_features import detect_circle_features, detect_lines
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


class IBVSController(Node):
    def __init__(self):
        super().__init__("ibvs_controller")

        # Create subscriber to the image topic
        self.subscription = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "camera/depth_image", self.depth_callback, 10
        )

        # Initialize depth image
        self.depth_image = None

        # Create publisher for velocity commands
        self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Initialize the OpenCV bridge
        self.bridge = CvBridge()

        # Create a callback group for the service client
        # self.callback_group = ReentrantCallbackGroup()

        # Create service server - this is what you'll call from terminal or other nodes
        self.srv = self.create_service(
            SetBool, "enable_ibvs", self.enable_ibvs_callback
        )

        # Parameters
        self.declare_parameter("visualize", True)  # Whether to show visualization
        self.declare_parameter("lambda_gain", 0.005)  # Control gain
        self.declare_parameter(
            "convergence_threshold", 10.0
        )  # Threshold for error convergence (pixels)
        self.declare_parameter(
            "min_circle_radius", 10
        )  # Minimum radius for circle detection
        self.declare_parameter(
            "max_circle_radius", 60
        )  # Maximum radius for circle detection

        # IBVS control enabled by default
        self.ibvs_enabled = False

        # Get parameters
        self.visualize = self.get_parameter("visualize").value
        self.lambda_gain = self.get_parameter("lambda_gain").value
        self.convergence_threshold = self.get_parameter("convergence_threshold").value
        self.min_circle_radius = self.get_parameter("min_circle_radius").value
        self.max_circle_radius = self.get_parameter("max_circle_radius").value

        # Camera intrinsic parameters (should be updated with your camera values)
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

        



        # Desired image points (target positions in the image)

        # Point 1: [184.00, 52.00] Depth: 1.17
        # Point 2: [405.00, 55.00] Depth: 1.19
        # Point 3: [184.00, 274.00] Depth: 1.17
        # Point 4: [405.00, 274.00] Depth: 1.19
        self.target_distance = np.array([1.28, 1.28, 1.28, 1.28])
        self.p_desired = np.array(
            [
                [218, 69],  # Top left
                [420, 69],  # Top right
                [218, 271],  # Bottom left
                [420, 271],  # Bottom right
            ],
            dtype=np.float32,
        )

        # Initialize current points
        self.p_current = None

        # Debug window
        if self.visualize:
            self.window_name = "IBVS Controller"
            cv2.namedWindow(self.window_name)

        self.get_logger().info("IBVS Controller initialized")

    def enable_ibvs_callback(self, request, response):
        """Service callback to enable/disable IBVS control"""
        self.ibvs_enabled = request.data

        if self.ibvs_enabled:
            self.get_logger().info("IBVS control enabled")
            response.message = "IBVS control enabled"
        else:
            self.get_logger().info("IBVS control disabled")
            # Send zero velocity to stop the robot
            self.stop_robot()
            response.message = "IBVS control disabled"

        response.success = True
        return response

    def depth_callback(self, msg=Image):
        """Callback for the depth image."""

        # Convert ROS Image message to OpenCV image, with encoding "32FC1"
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="32FC1"
        )  # Float format -> in meters

    def image_callback(self, msg=CompressedImage):
        try:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite("/home/tafarrel/handrail_test.jpg", cv_image)


            # detect lines using LSD
            # lines = detect_lines(cv_image)

            # Detect the 4 circle features using SIFT
            points = detect_circle_features(
                cv_image,
                self.p_desired,
                self.min_circle_radius,
                self.max_circle_radius,
                visualize=True,
            )

            
            # If we have detected the 4 points, update current points and perform IBVS
            if points is not None and len(points) == 4 and self.ibvs_enabled:
                self.p_current = points

                
                # Calculate and publish velocity commands
                self.calculate_velocity()


                # Visualize
                if self.visualize:
                    Z = np.array(
                        [
                            self.depth_image[int(point[1]), int(point[0])]
                            for point in self.p_current
                        ]
                    )

                    # print each point coordinates with the depth with ros log
                    # for i, point in enumerate(self.p_current):
                    #     self.get_logger().info(
                    #         f"Point {i+1}: [{point[0]:.2f}, {point[1]:.2f}] Depth: {Z[i]:.2f}"
                    #     )

                    self.display_visualization(cv_image)
            else:
                # No points detected
                self.stop_robot()
                if self.visualize:
                    cv2.putText(
                        cv_image,
                        "4 circles not detected",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(self.window_name, cv_image)
                    cv2.waitKey(1)
                if not self.ibvs_enabled:
                    self.get_logger().warn("IBVS control is disabled")
                else:
                    self.get_logger().warn("Could not detect 4 circle features")

        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            self.stop_robot()

    def calculate_interaction_matrix(self, points, Z):
        """Calculate the interaction matrix (image Jacobian) for the points."""
        L = np.zeros((2 * len(points), 6))

        for i, point in enumerate(points):
            x, y = point

            # Normalized image coordinates
            x_n = (x - self.K[0, 2]) / self.K[0, 0]
            y_n = (y - self.K[1, 2]) / self.K[1, 1]

            # Fill the interaction matrix for this point
            row = 2 * i
            # For vx
            L[row, 0] = -1.0 / Z[i]
            L[row, 1] = 0.0
            L[row, 2] = x_n / Z[i]
            L[row, 3] = x_n * y_n
            L[row, 4] = -(1.0 + x_n * x_n)
            L[row, 5] = y_n

            # For vy
            L[row + 1, 0] = 0.0
            L[row + 1, 1] = -1.0 / Z[i]
            L[row + 1, 2] = y_n / Z[i]
            L[row + 1, 3] = 1.0 + y_n * y_n
            L[row + 1, 4] = -x_n * y_n
            L[row + 1, 5] = -x_n

        return L

    def calculate_velocity(self):
        """Calculate and publish velocity commands based on visual error."""
        if self.p_current is None:
            return

        # Compute the error vector (current points - desired points)
        error = self.p_current.flatten() - self.p_desired.flatten()
        error_norm = np.linalg.norm(error)

        # Check if we're already converged
        if error_norm < self.convergence_threshold:
            self.get_logger().info(
                f"Converged: error = {error_norm:.2f} (< {self.convergence_threshold})"
            )
            self.stop_robot()
            return

        # Compute the depth of the target from the depth image and the center of each points
        Z = np.array(
            [self.depth_image[int(point[1]), int(point[0])] for point in self.p_current]
        )

        # Compute interaction matrix
        # We assume constant Z for simplicity, but this should be estimated from the actual scenario
        L = self.calculate_interaction_matrix(self.p_current, Z)

        # Compute pseudo-inverse of interaction matrix
        L_pinv, return_rank = pinv(L, return_rank=True)

        # Compute velocity command
        v = -self.lambda_gain * np.dot(L_pinv, error)

        # convert the image velocity axis to robot velocity axis, z_pixel -> x , -x_pixel -> y,  -y_pixel -> z
        v = np.array([v[2], -v[0], -v[1], v[5], -v[3], -v[4]])

        # Create and publish twist message
        twist = Twist()
        # Linear velocity
        twist.linear.x = v[0]
        twist.linear.y = v[1]
        twist.linear.z = v[2]
        # Angular velocity
        twist.angular.x = v[3]
        twist.angular.y = v[4]
        twist.angular.z = v[5]

        # Limit velocities for safety
        max_lin_vel = 0.2  # m/s
        max_ang_vel = 0.1  # rad/s

        # Limit each velocity component
        twist.linear.x = self.clamp(twist.linear.x, -max_lin_vel, max_lin_vel)
        twist.linear.y = self.clamp(twist.linear.y, -max_lin_vel, max_lin_vel)
        twist.linear.z = self.clamp(twist.linear.z, -max_lin_vel, max_lin_vel)
        twist.angular.x = self.clamp(twist.angular.x, -max_ang_vel, max_ang_vel)
        twist.angular.y = self.clamp(twist.angular.y, -max_ang_vel, max_ang_vel)
        twist.angular.z = self.clamp(twist.angular.z, -max_ang_vel, max_ang_vel)

        self.get_logger().info(
            f"Velocity x: {twist.linear.x}, y: {twist.linear.y}, rank : {return_rank}"
        )

        # Publish the velocity
        self.vel_publisher.publish(twist)

        # self.get_logger().info(f'Published velocity: lin=[{twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}], ' +
        #                       f'ang=[{twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f}], error={error_norm:.2f}')

    def stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        twist = Twist()
        self.vel_publisher.publish(twist)

    def clamp(self, value, min_value, max_value):
        """Clamp a value between min and max."""
        return max(min(value, max_value), min_value)

    def display_visualization(self, image):
        """Display visualization of the IBVS controller."""
        # Create a copy for visualization
        viz_img = image.copy()

        # Draw the current points
        for i, point in enumerate(self.p_current):
            cv2.circle(viz_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                viz_img,
                f"{i+1}",
                (int(point[0]) + 10, int(point[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw the desired points
        # for i, point in enumerate(self.p_desired):
        #     cv2.circle(viz_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        #     cv2.putText(viz_img, f"{i+1}'", (int(point[0])+10, int(point[1])+10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw lines between current and desired points
        # for i in range(4):
        #     p1 = (int(self.p_current[i][0]), int(self.p_current[i][1]))
        #     p2 = (int(self.p_desired[i][0]), int(self.p_desired[i][1]))
        #     cv2.line(viz_img, p1, p2, (255, 0, 0), 2)

        # Show error information
        error = np.linalg.norm(self.p_current.flatten() - self.p_desired.flatten())
        cv2.putText(
            viz_img,
            f"Error: {error:.2f} px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Display the image
        cv2.imshow(self.window_name, viz_img)
        # cv2.imshow("depth image",self.depth_image)
        cv2.waitKey(1)

    def destroy_node(self):
        """Clean up when the node is shutting down."""
        # Stop the robot
        self.stop_robot()

        # Close all OpenCV windows
        if self.visualize:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    # run a node to spawn the robot

    ibvs_controller = IBVSController()
    # executor = MultiThreadedExecutor()
    # executor.add_node(ibvs_controller)

    try:
        rclpy.spin(ibvs_controller)
        # spawn_robot_node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # executor.shutdown()
        ibvs_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
