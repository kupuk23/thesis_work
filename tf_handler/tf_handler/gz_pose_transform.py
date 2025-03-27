#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


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

        self.pointcloud_subscribe = self.create_subscription(
            PointCloud2, "/camera/points", self.pointcloud_callback, 10
        )

        # Set up the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store the handrail pose
        self.handrail_pose = None

        self.get_logger().info("Gazebo Pose Extractor Node has been initialized")

    def pointcloud_callback(self, msg):
        self.timestamp = msg.header.stamp

    def world_pose_callback(self, msg):
        """
        Process incoming world pose message and extract the pose of the handrail

        Args:
            msg: PoseArray message containing the world pose
        """
        try:
            if len(msg.poses) == 0:
                self.get_logger().info("No poses received")
                return

            # Extract the pose of the handrail
            self.handrail_pose = msg.poses[1]  # handrail is the second object

            self.grapple_pose = msg.poses[2]  # grapple is the third object

            # # Publish the extracted pose as a PoseStamped message
            # pose_stamped = PoseStamped()
            # pose_stamped.header.stamp = self.timestamp
            # pose_stamped.header.frame_id = "map"
            # pose_stamped.pose = self.handrail_pose

            # self.handrail_pose_publisher.publish(pose_stamped)

            # self.get_logger().info(f"Timestamp: {timestamp}")
            # Broadcast the transform to the TF tree
            self.broadcast_transform(
                self.handrail_pose, self.grapple_pose, self.timestamp
            )

        except Exception as e:
            self.get_logger().warn(f"Error broadcasting transform: {e}")

    def broadcast_transform(self, handrail_pose, grapple_pose, timestamp):
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
        transform.transform.translation.x = handrail_pose.position.x
        transform.transform.translation.y = handrail_pose.position.y
        transform.transform.translation.z = handrail_pose.position.z

        # Set rotation
        transform.transform.rotation.x = handrail_pose.orientation.x
        transform.transform.rotation.y = handrail_pose.orientation.y
        transform.transform.rotation.z = handrail_pose.orientation.z
        transform.transform.rotation.w = handrail_pose.orientation.w

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

        # create grapple transform
        grapple_transform = TransformStamped()
        grapple_transform.header.stamp = timestamp
        grapple_transform.header.frame_id = "map"
        grapple_transform.child_frame_id = "grapple"

        grapple_transform.transform.translation.x = grapple_pose.position.x
        grapple_transform.transform.translation.y = grapple_pose.position.y
        grapple_transform.transform.translation.z = grapple_pose.position.z

        grapple_transform.transform.rotation.x = grapple_pose.orientation.x
        grapple_transform.transform.rotation.y = grapple_pose.orientation.y
        grapple_transform.transform.rotation.z = grapple_pose.orientation.z
        grapple_transform.transform.rotation.w = grapple_pose.orientation.w

        self.tf_broadcaster.sendTransform(grapple_transform)


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
