import rclpy
import rclpy.duration
from rclpy.node import Node
import numpy as np
import open3d as o3d
import copy
import cv2
from cv_bridge import CvBridge
import time
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, PoseStamped
import message_filters
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# from icp_testing.icp import align_pc
from pose_estimation.icp_testing.icp import align_pc, draw_pose_axes, align_pc_o3d
from pose_estimation.tools.visualizer import visualize_point_cloud
from pose_estimation.tools.pose_estimation_tools import preprocess_model, filter_pc_background
from pose_estimation.tools.tf_utils import (
    pose_to_matrix,
    matrix_to_pose,
    transform_to_pose,
    apply_noise_to_transform
)


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        self.voxel_size = 0.005
        # preprocess_model("/home/tafarrel/", voxel_size=self.voxel_size)

        object = "grapple" # or "handrail"

        self.model_pcd = (
            o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/grapple_fixture_down.pcd")
            if object == "grapple"
            else o3d.io.read_point_cloud(
                "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd"
            )
        )
        
        self.model_handrail_pcd = o3d.io.read_point_cloud(
            "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd"
        )
        self.model_grapple_pcd = o3d.io.read_point_cloud(
            "/home/tafarrel/o3d_logs/grapple_fixture_down.pcd"
        )
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

        self.depth_sub = self.create_subscription(
            Image, "/camera/depth_image", self.depth_callback, 10
        )

        self.subscription = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.rgb_callback, 10
        )

        # Create subscribers for timeSync with pointcloud and pose
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/camera/points",
            self.pc2_callback,
            10,  # /camera/points <- CHANGE LATER!
        )

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=5))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for ICP result
        self.icp_result_pub = self.create_publisher(PoseStamped, "/pose/icp_result", 10)

        # Store the latest messages
        self.latest_pointcloud = None
        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.handrail_pose = None

        self.get_logger().info("Pose estimation node initialized")

    def pc2_callback(self, pointcloud_msg: PointCloud2):
        """
        Process pointcloud for ICP

        Args:
            pointcloud_msg: PointCloud2 message
        """

        try:
            # start_time = time.perf_counter()

            o3d_cloud = self.pc2_to_o3d_color(pointcloud_msg)
            # finished_time = time.perf_counter()

            # self.get_logger().info(
            #     f"Time taken to convert pointcloud: {finished_time - start_time}"
            # )

            # self.get_logger().info("Processing Pointcloud... ")
            scene_pcd = self.preprocess_pointcloud(o3d_cloud)
            # self.visualize_point_clouds(target=scene_pcd, target_filename="handrail_offset_right.pcd")

            # visualize_point_cloud(scene_pcd)

            # if scene_pcd empty, return
            if len(scene_pcd.points) == 0:
                self.get_logger().info("Scene point cloud is empty")
                return
            initial_transformation = self.transform_obj_pose(pointcloud_msg, "grapple")


            noisy_transformation = apply_noise_to_transform(initial_transformation, t_std=0.01, r_std=0.1) 

            # result = align_pc(self.model_handrail_pcd, scene_pcd, init_T=initial_transformation)
            result = align_pc_o3d(
                self.model_pcd,
                scene_pcd,
                init_T=noisy_transformation,
                voxel_size=self.voxel_size,
            )

            if result is None:
                # self.get_logger().info("ICP did not converge")
                return

            # T_camera_object = np.linalg.inv(result.transformation)

            # make a pose stamp and publish it
            pose_msg = PoseStamped()
            pose_msg.header.stamp = pointcloud_msg.header.stamp
            pose_msg.header.frame_id = pointcloud_msg.header.frame_id
            pose_msg.pose = matrix_to_pose(result.transformation)

            self.icp_result_pub.publish(pose_msg)

            # # # This is the object's pose in camera coordinates
            # position = T_camera_object[:3, 3]  # Translation vector
            # rotation = T_camera_object[:3, :3]  # Rotation matrix

            # predicted_image_pose = draw_pose_axes(
            #     self.cv_image, rotation, position, self.K
            # )

            # # Display the image with the pose
            # cv2.imshow("Pose Estimation", predicted_image_pose)
            # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error processing point cloud: {e}")

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
            #wait for transform
            if not (self.tf_buffer.can_transform(pc2_msg.header.frame_id, "map", rclpy.time.Time()) and self.tf_buffer.can_transform("map", obj_frame, rclpy.time.Time())):
                
                return None
            # Get the transform from world to camera
            map_T_cam = self.tf_buffer.lookup_transform(
                pc2_msg.header.frame_id,  # target frame
                "map",  # source frame (camera frame)
                rclpy.time.Time(),  # time
                timeout=rclpy.duration.Duration(seconds=1.0),
            )

            obj_T_map = self.tf_buffer.lookup_transform(
                "map",
                obj_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1),
            )

            obj_T_map = transform_to_pose(obj_T_map.transform)
            # Convert transform to matrix
            handrail_pose_in_camera_frame = tf2_geometry_msgs.do_transform_pose(
                obj_T_map, map_T_cam
            )
            # self.get_logger().info("Transformed handrail pose to camera frame")

            result_msg = PoseStamped()
            result_msg.header.stamp = pc2_msg.header.stamp
            result_msg.header.frame_id = pc2_msg.header.frame_id
            result_msg.pose = handrail_pose_in_camera_frame

            # self.icp_result_pub.publish(result_msg)

            # Convert pose to matrix
            handrail_pose_matrix = pose_to_matrix(handrail_pose_in_camera_frame)

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

        # filter maximum depth by z and x
        points_down = np.asarray(o3d_msg.points)
        colors_down = np.asarray(o3d_msg.colors)
        mask = (points_down[:, 2] > -0.7) & (points_down[:, 0] < 2)

        filtered_points = points_down[mask]
        filtered_colors = colors_down[mask]

        o3d_msg.points = o3d.utility.Vector3dVector(filtered_points)
        o3d_msg.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Estimate normals
        # scene_pcd_down.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #         radius=voxel_size * 2, max_nn=30
        #     )
        # )

        self.get_logger().info(f"Pointcloud processed: {len(o3d_msg.points)} points")

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
        rgb_floats = data[:, 4]  # float packed color (if offset=16 => col 4)

        # Convert float -> int for bitwise
        rgb_int = rgb_floats.view(np.int32)
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF

        # Combine & normalize
        colors = np.column_stack((r, g, b)).astype(np.float32) / 255.0

        # Find valid points (not NaN or inf)
        valid_idx = np.all(np.isfinite(xyz), axis=1)
        points = xyz[valid_idx]
        colors = colors[valid_idx]
        

        # visualize point cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        # cloud.paint_uniform_color([0, 1, 0.0])  # Gray for scene

        cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)

        # Remove background points

        filtered_cloud = filter_pc_background(cloud)
        filtered_cloud.paint_uniform_color([0, 0, 1])  # Gray for scene

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


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()
