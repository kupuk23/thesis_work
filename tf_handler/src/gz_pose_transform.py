#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from geometry_msgs.msg import PoseArray
import tf2_ros
from tf2_ros import TransformBroadcaster


class GazeboPoseExtractor(Node):
    """
    ROS2 Node that extracts an object pose from Gazebo based on the world pose info
    and broadcasts a world transform frame at the origin.
    """

    def __init__(self):
        super().__init__("gz_pose_transform")

        # Subscribe to the Gazebo world pose info topic
        self.subscription = self.create_subscription(
            PoseArray, "/world/iss_world/pose/info", self.world_pose_callback, 10
        )

        # Create a publisher for the extracted handrail pose
        self.handrail_pose_publisher = self.create_publisher(
            PoseStamped, "/iss_world/handrail_pose", 10
        )

        # Set up the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store the handrail pose
        self.handrail_pose = None

        self.get_logger().info("Gazebo Pose Extractor Node has been initialized")

    def world_pose_callback(self, msg):
        """
        Process incoming world pose message and extract the pose of the handrail

        Args:
            msg: PoseArray message containing the world pose
        """

        if len(msg.poses) == 0:
            self.get_logger().info("No poses received")
            return

        # Extract the pose of the handrail
        self.handrail_pose = msg.poses[
            1
        ]  # the handrail is the second object in the list

        # Publish the extracted pose as a PoseStamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = self.handrail_pose

        self.handrail_pose_publisher.publish(pose_stamped)

        current_time = self.get_clock().now().to_msg()

        # Broadcast the transform to the TF tree
        self.broadcast_transform(self.handrail_pose, current_time)

        # self.get_logger().debug(f"Handrail pose extracted: x={self.handrail_pose.position.x}, " +
        #                       f"y={self.handrail_pose.position.y}, z={self.handrail_pose.position.z}")

    def broadcast_transform(self, pose, timestamp):
        """
        Broadcast the pose as a transform to the TF tree

        Args:
            pose: Pose message containing position and orientation
            timestamp: Timestamp to use for the transform
        """
        transform = TransformStamped()

        # Set header information
        transform.header.stamp = timestamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "handrail"

        # Set translation
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z

        # Set rotation
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboPoseExtractor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly")
    except Exception as e:
        node.get_logger().error(f"Error in node: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
