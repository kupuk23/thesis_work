#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool
import cv2
import numpy as np
from rclpy.executors import MultiThreadedExecutor
# Import our PnP tracker class
from icp_testing.pnp_tracker import PnPTracker
from icp_testing.pnp_tracker_ui import PnPTrackerUI

class PnPTrackerNode(Node):
    def __init__(self):
        super().__init__('pnp_tracker_node')
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize PnP tracker
        # You can provide your camera parameters here if they're different
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )
        

        # reference corner points for PnP
        self.corners = np.array([
            [0.0, 0.0],
            [335.0, 0.0],
            [0.0, 339.0],
            [335.0, 339.0],
        ])

        self.tracker = PnPTracker(self.K)
        self.ui_handler = PnPTrackerUI(self.tracker)
        self.ui_handler.start()  # Start the UI thread

        
        self.srv = self.create_service(
            SetBool, "enable_selection", self.enable_selection_callback
        )
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
            )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/object_pose',
            10)
        
        self.viz_pub = self.create_publisher(
            Image,
            '/tracking_visualization',
            10)
        
        # Setup for interactive mode if needed
        self.selection_mode = False
        self.create_timer(0.1, self.selection_timer_callback)
        
        self.get_logger().info('PnP Tracker Node initialized')
    
    def enable_selection_callback(self, request, response):
        """Callback for the service to enable object selection."""
        if request.data:
            self.get_logger().info('Starting object selection')
            self.start_selection_mode(self.cv_image)
        response.success = True
        response.message = 'Object selection enabled'
        return response

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.ui_handler.update_frame(self.cv_image)

            # Check if we're in selection mode
            if self.selection_mode:
                # We're handling selection in the timer callback
                return
            
            # Process the frame for tracking
            if self.tracker.track_mode:
                result_frame, translation, rotation = self.tracker.process_frame(self.cv_image)

                # display result frame
                # cv2.imshow('Result Frame', result_frame)
                # cv2.waitKey(1)
                
                # Publish tracking visualization
                viz_msg = self.bridge.cv2_to_imgmsg(result_frame, encoding='bgr8')
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)
                
                # If tracking was successful, publish the pose
                if translation is not None and rotation is not None:
                    pose_msg = self.create_pose_message(translation, rotation, msg.header.stamp)
                    self.pose_pub.publish(pose_msg)
        
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')
    
    def create_pose_message(self, translation, rotation, timestamp):
        """Create a PoseStamped message from the tracking results."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = 'camera_frame'
        
        # Set position (convert to meters)
        pose_msg.pose.position.x = float(translation[0][0]) / 100.0  # cm to m
        pose_msg.pose.position.y = float(translation[1][0]) / 100.0
        pose_msg.pose.position.z = float(translation[2][0]) / 100.0
        
        # Convert rotation from Euler angles to quaternion
        # Note: This assumes the rotation is in the correct order (RPY)
        # You might need to adjust the conversion depending on your requirements
        roll = float(rotation[0][0]) * np.pi / 180.0
        pitch = float(rotation[1][0]) * np.pi / 180.0
        yaw = float(rotation[2][0]) * np.pi / 180.0
        
        # Convert Euler angles to quaternion (simplified version)
        # For a more accurate conversion, consider using transformations library
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        pose_msg.pose.orientation.w = cr * cp * cy + sr * sp * sy
        pose_msg.pose.orientation.x = sr * cp * cy - cr * sp * sy
        pose_msg.pose.orientation.y = cr * sp * cy + sr * cp * sy
        pose_msg.pose.orientation.z = cr * cp * sy - sr * sp * cy
        
        return pose_msg
    
    def selection_timer_callback(self):
        """Handle the interactive selection mode."""
        if not self.selection_mode:
            return
        
        # Check if selection is complete
        if self.tracker.check_object_selection_complete():
            self.selection_mode = False
            self.get_logger().info('Object selection complete')
            return
    
    def start_selection_mode(self, frame):
        """Start the object selection mode."""
        self.selection_mode = True
        display_frame = self.tracker.start_object_selection(frame)
        
        # Setup a window for selection
        cv2.namedWindow('Object Selection')
        # cv2.setMouseCallback('Object Selection', self.mouse_callback)
        
        # Display the frame
        cv2.imshow('Object Selection', display_frame)
        cv2.waitKey(1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for object selection."""
        if self.tracker.mouse_callback(event, x, y, flags, param):
            # Update the display
            cv2.imshow('Object Selection', self.tracker.current_frame)
            cv2.waitKey(1)

    def destroy_node(self):
        # Clean up UI thread when the node is destroyed
        if hasattr(self, 'ui_handler'):
            self.ui_handler.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PnPTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()