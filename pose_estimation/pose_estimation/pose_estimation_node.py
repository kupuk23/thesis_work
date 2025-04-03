import rclpy
import rclpy.duration
from rclpy.node import Node
import numpy as np
import open3d as o3d
import copy
import cv2
from cv_bridge import CvBridge
import time
import rclpy.time
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf2_geometry_msgs
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)


# from icp_testing.icp import align_pc
from pose_estimation.icp_testing.icp import align_pc_o3d
from pose_estimation.tools.visualizer import visualize_point_cloud
from pose_estimation.tools.pose_estimation_tools import (
    preprocess_model,
    filter_pc_background,
)
from pose_estimation.tools.tf_utils import (
    pose_to_matrix,
    matrix_to_pose,
    transform_to_pose,
    apply_noise_to_transform,
)


class PoseEstimationNode(Node):
    def __init__(self, qos_profile=None):
        super().__init__("pose_estimation_node")

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        self.voxel_size = 0.005
        # preprocess_model("/home/tafarrel/", voxel_size=self.voxel_size)

        self.object = "grapple"  # or "handrail"

        self.model_pcd = (
            o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/grapple_fixture_down.pcd")
            if self.object == "grapple"
            else o3d.io.read_point_cloud(
                "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd"
            )
        )

        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

        # self.depth_sub = self.create_subscription(
        #     Image, "/camera/depth_image", self.depth_callback, 10
        # )

        # self.subscription = self.create_subscription(
        #     CompressedImage, "/camera/image/compressed", self.rgb_callback, 10
        # )

        # Create subscribers for timeSync with pointcloud and pose
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/camera/points",
            self.pc2_callback,
            qos_profile=20,  # /camera/points <- CHANGE LATER!
            callback_group=ReentrantCallbackGroup(),
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        mutex_group = MutuallyExclusiveCallbackGroup()
        # Create a timer for transform updates that runs at high frequency
        self.process_pose_estimation_timer = self.create_timer(
            0.1, self.perform_pose_estimation, callback_group=mutex_group
        )

        # Publisher for ICP result
        self.icp_result_pub = self.create_publisher(PoseStamped, "/pose/icp_result", 10)

        # Store the latest messages
        self.latest_pointcloud = None
        self.latest_transform_result = None
        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.handrail_pose = None

        self.get_logger().info("Pose estimation node initialized")

    def perform_pose_estimation(self):

        if self.latest_transform_result is None:
            self.get_logger().info("No transform available")
            return
        try:
            start_time = time.perf_counter()

            o3d_cloud = self.pc2_to_o3d_color(self.latest_pointcloud)
            finished_time = time.perf_counter()

            self.get_logger().info(
                f"Pointcloud retrieved from camera.. ({finished_time - start_time}s)"
            )

            # self.get_logger().info("Processing Pointcloud... ")

            start_time = time.perf_counter()
            scene_pcd = self.preprocess_pointcloud(o3d_cloud)
            finished_time = time.perf_counter()
            self.get_logger().info(
                f"Pointcloud processed --> {len(scene_pcd.points)} points ({finished_time - start_time}s)"
            )

            # if scene_pcd empty, return
            if len(scene_pcd.points) < 100:
                self.get_logger().info("Scene point cloud is empty")
                return

            noisy_transformation = apply_noise_to_transform(
                self.latest_transform_result, t_std=0.025, r_std=0.25
            )  # t_std=0.025, r_std=0.25

            start_time = time.perf_counter()
            result = align_pc_o3d(
                self.model_pcd,
                scene_pcd,
                init_T=noisy_transformation,
                voxel_size=self.voxel_size,
            )
            finished_time = time.perf_counter()
            self.get_logger().info(f"ICP finished --> ({finished_time - start_time}s)")

            if result is None:
                # self.get_logger().info("ICP did not converge")
                # remove icp_result publisher
                empty_pose = PoseStamped()
                empty_pose.header.stamp = self.latest_pointcloud.header.stamp
                empty_pose.header.frame_id = "map"
                t = np.zeros((4, 4))
                t[0:3, 0:3] = np.eye(3)
                empty_pose.pose = matrix_to_pose(t)
                self.icp_result_pub.publish(empty_pose)
                return

            # T_camera_object = np.linalg.inv(result.transformation)

            # make a pose stamp and publish it
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.latest_pointcloud.header.stamp
            pose_msg.header.frame_id = self.latest_pointcloud.header.frame_id
            pose_msg.pose = matrix_to_pose(result.transformation)

            self.icp_result_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().info(f"Error processing point cloud: {e}")

    def pc2_callback(self, pointcloud_msg: PointCloud2):
        """
        Process pointcloud for ICP

        Args:
            pointcloud_msg: PointCloud2 message
        """
        self.latest_pointcloud = pointcloud_msg
        self.update_transform()
        # self.get_logger().info(f"Received pointcloud")

    def transform_obj_pose(self, pc2_msg: PointCloud2, obj_frame="handrail"):
        """
        Transform the obj_pose stamped (parent frame : map) to camera frame
        Args:
            obj_pose_stamped: PoseStamped message with object pose in map frame
            frame_id: Camera frame id
        Returns:
            handrail_pose_matrix: Object pose in camera frame as matrix
        """
        try:
            # wait for transform
            if not self.tf_buffer.can_transform(
                pc2_msg.header.frame_id, obj_frame, rclpy.time.Time()
            ):
                return None

            # Get the transform from obj to camera

            # map_T_cam = self.tf_buffer.lookup_transform(
            #     pc2_msg.header.frame_id,  # target frame
            #     "map",  # source frame (camera frame)
            #     rclpy.time.Time(),  # latest available transform
            #     timeout=rclpy.duration.Duration(seconds=2),
            # )
            obj_T_cam = self.tf_buffer.lookup_transform(
                pc2_msg.header.frame_id,
                obj_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            obj_T_cam = transform_to_pose(obj_T_cam.transform)

            # self.get_logger().info("Transformed handrail pose to camera frame")

            result_msg = PoseStamped()
            result_msg.header.stamp = pc2_msg.header.stamp
            result_msg.header.frame_id = pc2_msg.header.frame_id
            result_msg.pose = obj_T_cam

            self.icp_result_pub.publish(result_msg)

            # Convert pose to matrix
            handrail_pose_matrix = pose_to_matrix(
                obj_T_cam
            )  # pose_to_matrix(handrail_pose_in_camera_frame)

            return handrail_pose_matrix

        except Exception as e:
            self.get_logger().info(f"Error transforming handrail pose : {e}")

    def preprocess_pointcloud(self, o3d_msg):
        """
        Preprocess the pointcloud from ROS topic

        Args:
            pointcloud_msg: ROS PointCloud2 message

        Returns:
            Preprocessed scene point cloud as Open3D point cloud
        """

        # Remove statistical outliers

        # o3d_msg, _ = o3d_msg.remove_statistical_outlier(
        #     nb_neighbors=20, std_ratio=2.0
        # )

        if len(o3d_msg.colors) == 0:
            # self.get_logger().info("No colors in point cloud")
            color = False

        # filter maximum depth by z and x
        points_down = np.asarray(o3d_msg.points)
        mask = (points_down[:, 2] > -0.7) & (points_down[:, 0] < 2)

        filtered_points = points_down[mask]

        if color:
            colors_down = np.asarray(o3d_msg.colors)
            filtered_colors = colors_down[mask]
            o3d_msg.colors = o3d.utility.Vector3dVector(filtered_colors)

        o3d_msg.points = o3d.utility.Vector3dVector(filtered_points)

        # self.visualize_point_clouds(source=self.model_handrail_pcd, target=scene_pcd_down)
        return o3d_msg

    def depth_callback(self, msg):
        """Process incoming depth image"""
        self.latest_depth_image = msg

        # Convert depth image to point cloud if needed
        # Note: This is an alternative to using the pointcloud topic
        if self.latest_pointcloud is None:  # Only use if pointcloud isn't available
            # Convert ROS image to OpenCV image
            depth_image = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )

            # Here you would convert depth to pointcloud using camera intrinsics
            # This would require camera intrinsic parameters
            # For now we'll skip this as we're using the direct pointcloud

    def pc2_to_o3d_color(self, pc2_msg):
        """
        Convert ROS PointCloud2 message to downsapled Open3D point cloud
        input: PointCloud2 message
        output: Open3D point cloud (downsampled)

        """
        n_points = pc2_msg.width * pc2_msg.height
        if n_points == 0:
            return o3d.geometry.PointCloud()

        # Each 'float32' is 4 bytes. For example, if point_step=24 -> 24/4 = 6 floats/point
        floats_per_point = pc2_msg.point_step // 4

        # Parse all raw data in one shot
        data = np.frombuffer(pc2_msg.data, dtype=np.float32)
        data = data.reshape(n_points, floats_per_point)

        # Adjust these slices for your actual offsets:
        #   x = column 0
        #   y = column 1
        #   z = column 2
        #   (possible padding at column 3)
        #   rgb = column 4
        xyz = data[:, 0:3]

        # rgb_floats = data[:, 4]  # float packed color (if offset=16 => col 4)

        # # Convert float -> int for bitwise
        # rgb_int = rgb_floats.view(np.int32)
        # r = (rgb_int >> 16) & 0xFF
        # g = (rgb_int >> 8) & 0xFF
        # b = rgb_int & 0xFF

        # # Combine & normalize
        # colors = np.column_stack((r, g, b)).astype(np.float32) / 255.0

        # Find valid points (not NaN or inf)
        valid_idx = np.all(np.isfinite(xyz), axis=1)
        points = xyz[valid_idx]
        # colors = colors[valid_idx]

        # visualize point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        # cloud.colors = o3d.utility.Vector3dVector(colors)
        # cloud.paint_uniform_color([0, 1, 0.0])  # Gray for scene

        cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)

        # Remove background points

        # filtered_cloud = filter_pc_background(cloud)
        # filtered_cloud.paint_uniform_color([0, 0, 1])  # Gray for scene

        # o3d.visualization.draw_geometries(
        #     [filtered_cloud])

        return cloud

    def rgb_callback(self, msg):
        """Process incoming RGB image"""
        try:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv2.imwrite("/home/tafarrel/handrail_offset_right.jpg", self.cv_image)
        except:
            self.get_logger().info("Error converting image")
        # Could be used for visualization or feature extraction from RGB

    def visualize_point_clouds(
        self, target, source=None, target_filename="target.pcd", transformed_source=None
    ):
        """
        Visualize the point clouds (non-blocking in separate process)

        Args:
            source: Source point cloud (model)
            target: Target point cloud (scene)
            transformed_source: Transformed source after registration (if available)
        """
        if source is not None:
            source_temp = copy.deepcopy(source)
            source_temp.paint_uniform_color([1, 0, 0])  # Red for model
            o3d.io.write_point_cloud("/home/tafarrel/o3d_logs/source.pcd", source_temp)

        # Create visualization geometries

        target_temp = copy.deepcopy(target)

        # Color the point clouds

        target_temp.paint_uniform_color([0, 1, 0])  # Green for scene

        # Save to file for later visualization (non-blocking)

        o3d.io.write_point_cloud(
            f"/home/tafarrel/o3d_logs/{target_filename}", target_temp
        )
        self.get_logger().info(
            "Point clouds saved to /home/tafarrel/o3d_logs/ directory for visualization"
        )

        # Note: Using draw_geometries directly would block the ROS node
        # For ROS, it's better to save the point clouds and visualize them separately
        # or use RViz for visualization

    def update_transform(self):
        try:
            # wait for transform
            if not self.tf_buffer.can_transform(
                self.latest_pointcloud.header.frame_id, self.object, rclpy.time.Time()
            ):
                return None

            # Get the transform from obj to camera

            # map_T_cam = self.tf_buffer.lookup_transform(
            #     pc2_msg.header.frame_id,  # target frame
            #     "map",  # source frame (camera frame)
            #     rclpy.time.Time(),  # latest available transform
            #     timeout=rclpy.duration.Duration(seconds=2),
            # )
            obj_T_cam = self.tf_buffer.lookup_transform(
                self.latest_pointcloud.header.frame_id,
                self.object,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            obj_T_cam = transform_to_pose(obj_T_cam.transform)

            # self.get_logger().info("Transformed handrail pose to camera frame")

            # self.icp_result_pub.publish(result_msg)

            transform = TransformStamped()
            transform.header.stamp = self.latest_pointcloud.header.stamp
            transform.header.frame_id = self.latest_pointcloud.header.frame_id
            transform.child_frame_id = "ground_truth_pose"
            transform.transform.translation.x = obj_T_cam.position.x
            transform.transform.translation.y = obj_T_cam.position.y
            transform.transform.translation.z = obj_T_cam.position.z
            transform.transform.rotation.x = obj_T_cam.orientation.x
            transform.transform.rotation.y = obj_T_cam.orientation.y
            transform.transform.rotation.z = obj_T_cam.orientation.z
            transform.transform.rotation.w = obj_T_cam.orientation.w

            # broadcast the transform
            self.tf_broadcaster.sendTransform(transform)
            obj_T_cam = pose_to_matrix(obj_T_cam)
            self.latest_transform_result = obj_T_cam
            # return handrail_pose_matrix

        except Exception as e:
            self.get_logger().info(f"Error transforming handrail pose : {e}")


def main(args=None):
    rclpy.init(args=args)
    # qos_profile = QoSProfile(
    #     depth=20,
    #     history=QoSHistoryPolicy.KEEP_LAST,
    #     reliability=QoSReliabilityPolicy.BEST_EFFORT,
    #     durability=QoSDurabilityPolicy.VOLATILE,
    # )
    node = PoseEstimationNode()
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":

    main()
