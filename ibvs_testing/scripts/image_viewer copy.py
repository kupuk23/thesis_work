#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.linalg import pinv


class IBVSController(Node):
    def __init__(self):
        super().__init__("ibvs_controller")

        # Create subscriber to the image topic
        self.subscription = self.create_subscription(
            Image, "camera/image", self.image_callback, 10
        )

        # create subscriber to depth image
        self.depth_subscription = self.create_subscription(
            Image, "camera/depth", self.depth_callback, 10
        )

        # Create publisher for velocity commands
        self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Initialize the OpenCV bridge
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter("visualize", True)  # Whether to show visualization
        self.declare_parameter("lambda_gain", 0.2)  # Control gain
        self.declare_parameter("target_distance", 1.0)  # Desired z distance to target
        self.declare_parameter(
            "convergence_threshold", 5.0
        )  # Threshold for error convergence (pixels)

        # Get parameters
        self.visualize = self.get_parameter("visualize").value
        self.lambda_gain = self.get_parameter("lambda_gain").value
        self.target_distance = self.get_parameter("target_distance").value
        self.convergence_threshold = self.get_parameter("convergence_threshold").value

        # Camera intrinsic parameters (should be updated with your camera values)
        self.K = np.array(
            [
                [221.76500407999384, 0.0, 160.0],  # fx, 0, cx
                [0.0, 221.76500407999382, 120.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

        self.depth_image = None

        # Desired image points (target positions in the image)
        # Default: four points forming a square in the center
        size = 100
        cx, cy = 320, 240  # Image center
        self.p_desired = np.array(
            [
                [cx - size, cy - size],  # Top left
                [cx + size, cy - size],  # Top right
                [cx + size, cy + size],  # Bottom right
                [cx - size, cy + size],  # Bottom left
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

    def depth_callback(self, msg):

        # Convert ROS Image message to OpenCV image
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Detect the 4 black points
            points = self.detect_points(cv_image)

            # If we have detected the 4 points, update current points and perform IBVS
            if points is not None and len(points) == 4:
                self.p_current = points

                # Calculate and publish velocity commands
                self.calculate_velocity()

                # Visualize
                if self.visualize:
                    self.display_visualization(cv_image)
            else:
                # No points detected
                self.stop_robot()
                if self.visualize:
                    cv2.putText(
                        cv_image,
                        "Points not detected",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(self.window_name, cv_image)
                    cv2.waitKey(1)
                self.get_logger().warn("Could not detect 4 points")

        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            self.stop_robot()

    def detect_points(self, img):
        """Detect 4 black points in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to isolate dark points
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area to find the points
        min_area = 20  # Minimum area for a valid point
        max_area = 500  # Maximum area for a valid point

        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))

        # We need exactly 4 points
        if len(centers) != 4:
            return None

        # Sort points to match the desired order (top-left, top-right, bottom-right, bottom-left)
        # First sort by y-coordinate (top to bottom)
        centers = sorted(centers, key=lambda p: p[1])

        # Get top two and bottom two points
        top_points = sorted(centers[:2], key=lambda p: p[0])  # Left to right
        bottom_points = sorted(centers[2:], key=lambda p: p[0])  # Left to right

        # Combine them in the correct order
        ordered_points = np.array(top_points + bottom_points, dtype=np.float32)

        return ordered_points

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
            L[row, 0] = -1.0 / Z
            L[row, 1] = 0.0
            L[row, 2] = x_n / Z
            L[row, 3] = x_n * y_n
            L[row, 4] = -(1.0 + x_n * x_n)
            L[row, 5] = y_n

            # For vy
            L[row + 1, 0] = 0.0
            L[row + 1, 1] = -1.0 / Z
            L[row + 1, 2] = y_n / Z
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

        # Compute interaction matrix
        # We assume constant Z for simplicity, but this should be estimated from the actual scenario
        L = self.calculate_interaction_matrix(self.p_current, self.target_distance)

        # Compute pseudo-inverse of interaction matrix
        L_pinv = pinv(L)

        # Compute velocity command
        v = -self.lambda_gain * np.dot(L_pinv, error)

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
        max_ang_vel = 0.3  # rad/s

        # limit the velocities
        for attr in ["x", "y", "z"]:
            setattr(
                twist.linear,
                attr,
                self.clamp(getattr(twist.linear, attr), -max_lin_vel, max_lin_vel),
            )
            setattr(
                twist.angular,
                attr,
                self.clamp(getattr(twist.angular, attr), -max_ang_vel, max_ang_vel),
            )

        # Publish the velocity
        self.vel_publisher.publish(twist)

        self.get_logger().info(
            f"Published velocity: lin=[{twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}], "
            + f"ang=[{twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f}], error={error_norm:.2f}"
        )

    def stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        twist = Twist()
        self.vel_publisher.publish(twist)

    def clamp(self, value, min_value, max_value):
        """Clamp a value between min and max."""
        return max(min(value, max_value), min_value)

    def display_visualization(self, image):
        """Display visualization of the IBVS controller."""
        # Draw the current points
        for i, point in enumerate(self.p_current):
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                image,
                f"{i+1}",
                (int(point[0]) + 10, int(point[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw the desired points
        for i, point in enumerate(self.p_desired):
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"{i+1}'",
                (int(point[0]) + 10, int(point[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # Draw lines between current and desired points
        for i in range(4):
            p1 = (int(self.p_current[i][0]), int(self.p_current[i][1]))
            p2 = (int(self.p_desired[i][0]), int(self.p_desired[i][1]))
            cv2.line(image, p1, p2, (255, 0, 0), 2)

        # Show error information
        error = np.linalg.norm(self.p_current.flatten() - self.p_desired.flatten())
        cv2.putText(
            image,
            f"Error: {error:.2f} px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Display the image
        cv2.imshow(self.window_name, image)
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

    ibvs_controller = IBVSController()

    try:
        rclpy.spin(ibvs_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        ibvs_controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
