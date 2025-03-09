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
        super().__init__('ibvs_controller')
        
        # Create subscriber to the image topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10
        )
        
        # Create publisher for velocity commands
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vell',
            10
        )
        
        # Initialize the OpenCV bridge
        self.bridge = CvBridge()
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Parameters
        self.declare_parameter('visualize', True)  # Whether to show visualization
        self.declare_parameter('lambda_gain', 0.5)  # Control gain
        self.declare_parameter('target_distance', 0.5)  # Desired z distance to target
        self.declare_parameter('convergence_threshold', 5.0)  # Threshold for error convergence (pixels)
        self.declare_parameter('min_circle_radius', 10)  # Minimum radius for circle detection
        self.declare_parameter('max_circle_radius', 50)  # Maximum radius for circle detection
        
        # Get parameters
        self.visualize = self.get_parameter('visualize').value
        self.lambda_gain = self.get_parameter('lambda_gain').value
        self.target_distance = self.get_parameter('target_distance').value
        self.convergence_threshold = self.get_parameter('convergence_threshold').value
        self.min_circle_radius = self.get_parameter('min_circle_radius').value
        self.max_circle_radius = self.get_parameter('max_circle_radius').value
        
        # Camera intrinsic parameters (should be updated with your camera values)
        self.K = np.array([
            [500.0, 0.0, 320.0],  # fx, 0, cx
            [0.0, 500.0, 240.0],  # 0, fy, cy
            [0.0, 0.0, 1.0]       # 0, 0, 1
        ])
        
        # Desired image points (target positions in the image)
        # Default: four points forming a square in the center
        size = 100
        cx, cy = 320, 240  # Image center
        self.p_desired = np.array([
            [cx-size, cy-size],  # Top left
            [cx+size, cy-size],  # Top right
            [cx-size, cy+size],  # Bottom left
            [cx+size, cy+size]   # Bottom right
        ], dtype=np.float32)
        
        # Initialize current points
        self.p_current = None
        
        # Debug window
        if self.visualize:
            self.window_name = 'IBVS Controller'
            cv2.namedWindow(self.window_name)
        
        self.get_logger().info('IBVS Controller initialized with SIFT feature detection')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite('/home/tafarrel/image2.jpg', cv_image)
            
            # Detect the 4 circle features using SIFT
            points = self.detect_circle_features_sift(cv_image)
            
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
                    cv2.putText(cv_image, "4 circles not detected", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(self.window_name, cv_image)
                    cv2.waitKey(1)
                self.get_logger().warn('Could not detect 4 circle features')
                
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')
            self.stop_robot()

    def detect_circle_features_sift(self, img):
        """
        Detect 4 circle features using SIFT and return their centroids ordered from
        top-left to bottom-right.
        
        Args:
            img: Input image (BGR)
            
        Returns:
            numpy.ndarray: Array of 4 points ordered [top-left, top-right, bottom-left, bottom-right]
                           or None if 4 circles are not detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find circles
        circle_centers = []
        
        for contour in contours:
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Skip very small contours
            if area < np.pi * (self.min_circle_radius ** 2):
                continue
                
            # Skip very large contours
            if area > np.pi * (self.max_circle_radius ** 2):
                continue
            
            # Calculate circularity (4π × area / perimeter²)
            # A perfect circle has circularity = 1
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Filter for circular shapes
            if circularity > 0.5:  # Threshold for "circle-like" shapes
                # Calculate moments to find centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Apply SIFT to get more precise center if needed
                    roi = gray[max(0, cy-20):min(gray.shape[0], cy+20), 
                               max(0, cx-20):min(gray.shape[1], cx+20)]
                    
                    if roi.size > 0:  # Make sure ROI is not empty
                        keypoints = self.sift.detect(roi, None)
                        
                        # If SIFT finds keypoints in the ROI, refine the center
                        if keypoints:
                            strongest_kp = max(keypoints, key=lambda kp: kp.response)
                            refined_x = strongest_kp.pt[0] + max(0, cx-20)
                            refined_y = strongest_kp.pt[1] + max(0, cy-20)
                            circle_centers.append((refined_x, refined_y))
                        else:
                            # If SIFT fails, use the centroid from moments
                            circle_centers.append((cx, cy))
                    else:
                        circle_centers.append((cx, cy))
        
        # We need exactly 4 circles
        if len(circle_centers) != 4:
            self.get_logger().warn(f'Found {len(circle_centers)} circles instead of 4')
            
            # If visualization is enabled, draw the circles that were found
            if self.visualize:
                viz_img = img.copy()
                for center in circle_centers:
                    cv2.circle(viz_img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                cv2.putText(viz_img, f"Found {len(circle_centers)} circles", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Circle Detection", viz_img)
                cv2.waitKey(1)
            
            return None
        
        # Convert to numpy array
        centers = np.array(circle_centers, dtype=np.float32)
        
        # Order the points: top-left, top-right, bottom-left, bottom-right
        # First, sort by y-coordinate to separate top and bottom pairs
        centers = sorted(centers, key=lambda p: p[1])
        
        # Get top and bottom pairs
        top_pair = sorted(centers[:2], key=lambda p: p[0])      # Sort by x for top pair
        bottom_pair = sorted(centers[2:], key=lambda p: p[0])   # Sort by x for bottom pair
        
        # Combine into the final order: [top-left, top-right, bottom-left, bottom-right]
        ordered_centers = np.array([top_pair[0], top_pair[1], bottom_pair[0], bottom_pair[1]], dtype=np.float32)
        
        return ordered_centers

    def calculate_interaction_matrix(self, points, Z):
        """Calculate the interaction matrix (image Jacobian) for the points."""
        L = np.zeros((2*len(points), 6))
        
        for i, point in enumerate(points):
            x, y = point
            
            # Normalized image coordinates
            x_n = (x - self.K[0, 2]) / self.K[0, 0]
            y_n = (y - self.K[1, 2]) / self.K[1, 1]
            
            # Fill the interaction matrix for this point
            row = 2*i
            # For vx
            L[row, 0] = -1.0/Z
            L[row, 1] = 0.0
            L[row, 2] = x_n/Z
            L[row, 3] = x_n*y_n
            L[row, 4] = -(1.0 + x_n*x_n)
            L[row, 5] = y_n
            
            # For vy
            L[row+1, 0] = 0.0
            L[row+1, 1] = -1.0/Z
            L[row+1, 2] = y_n/Z
            L[row+1, 3] = 1.0 + y_n*y_n
            L[row+1, 4] = -x_n*y_n
            L[row+1, 5] = -x_n
            
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
            self.get_logger().info(f'Converged: error = {error_norm:.2f} (< {self.convergence_threshold})')
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
        
        # Limit each velocity component
        twist.linear.x = self.clamp(twist.linear.x, -max_lin_vel, max_lin_vel)
        twist.linear.y = self.clamp(twist.linear.y, -max_lin_vel, max_lin_vel)
        twist.linear.z = self.clamp(twist.linear.z, -max_lin_vel, max_lin_vel)
        twist.angular.x = self.clamp(twist.angular.x, -max_ang_vel, max_ang_vel)
        twist.angular.y = self.clamp(twist.angular.y, -max_ang_vel, max_ang_vel)
        twist.angular.z = self.clamp(twist.angular.z, -max_ang_vel, max_ang_vel)
        
        # Publish the velocity
        self.vel_publisher.publish(twist)
        
        self.get_logger().info(f'Published velocity: lin=[{twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}], ' +
                              f'ang=[{twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f}], error={error_norm:.2f}')

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
            cv2.putText(viz_img, f"{i+1}", (int(point[0])+10, int(point[1])+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw the desired points
        for i, point in enumerate(self.p_desired):
            cv2.circle(viz_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            cv2.putText(viz_img, f"{i+1}'", (int(point[0])+10, int(point[1])+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw lines between current and desired points
        for i in range(4):
            p1 = (int(self.p_current[i][0]), int(self.p_current[i][1]))
            p2 = (int(self.p_desired[i][0]), int(self.p_desired[i][1]))
            cv2.line(viz_img, p1, p2, (255, 0, 0), 2)
        
        # Show error information
        error = np.linalg.norm(self.p_current.flatten() - self.p_desired.flatten())
        cv2.putText(viz_img, f"Error: {error:.2f} px", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the image
        cv2.imshow(self.window_name, viz_img)
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

if __name__ == '__main__':
    main()