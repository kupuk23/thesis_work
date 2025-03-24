import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import copy
import cv2
from cv_bridge import CvBridge
import struct
import matplotlib.pyplot as plt
import time
import ctypes
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray

# from icp_testing.icp import align_pc
from icp_testing.pnp_tracker import PnPTracker
from pose_estimation.icp_testing.icp import align_pc, draw_pose_axes
from pose_estimation.tools.visualizer import visualize_point_cloud


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        self.voxel_size = 0.005

        # Load the model (only needs to be done once)
        # self.model_pcd = self.preprocess_model(
        #     "/home/tafarrel/blender_files/handrail/handrail.obj",
        #     voxel_size=0.005,
        # )

        self.model_pcd = o3d.io.read_point_cloud(
            "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd"
        )
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

        self.tracker = PnPTracker(self.K)

        # Create subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/asd",
            self.pointcloud_callback,
            10,  # /camera/points <- CHANGE LATER!
        )

        self.world_pose_sub = self.create_subscription(
            PoseArray, "/world/iss_world/pose/info", self.world_pose_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "/camera/depth_image", self.depth_callback, 10
        )

        self.subscription = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.rgb_callback, 10
        )

        # Store the latest messages
        self.latest_pointcloud = None
        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.handrail_pose = None

        self.get_logger().info("Pose estimation node initialized")

    def preprocess_model(self, model_path, voxel_size=0.01):
        """
        Preprocess the 3D model from Blender

        Args:
            model_path: Path to the model file (.obj, .stl, etc.)
            voxel_size: Downsampling voxel size

        Returns:
            Preprocessed model point cloud
        """
        self.get_logger().info(f"Loading model from {model_path}")

        # Load the model
        model = o3d.io.read_triangle_mesh(model_path)

        # Ensure model has normals
        model.compute_vertex_normals()

        # Sample points from the model surface
        model_pcd = model.sample_points_uniformly(number_of_points=10000)

        # Downsample the point cloud
        model_pcd_down = model_pcd.voxel_down_sample(voxel_size)

        # Estimate normals if they don't exist or need recalculation
        model_pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )

        # save the preprocessed model
        o3d.io.write_point_cloud(
            "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", model_pcd_down
        )

        self.get_logger().info(
            f"Model preprocessed: {len(model_pcd_down.points)} points"
        )
        return model_pcd_down

    def pointcloud_callback(self, msg):
        """Process incoming pointcloud data"""
        try:
            start_time = time.perf_counter()

            # xyz, rgb = self.pointcloud2_to_xyzrgb(msg)
            # o3d_cloud = o3d.geometry.PointCloud()
            # o3d_cloud.points = o3d.utility.Vector3dVector(xyz)
            # o3d_cloud.colors = o3d.utility.Vector3dVector(rgb)
            # self.get_logger().info(f"Point cloud received: {len(xyz)} points")
            # o3d_cloud, _, _= self.converter.ROSpc2_to_O3DPointCloud(msg)
            # self.get_logger().info(f"Point cloud received: {len(o3d_cloud.points)} points")

            o3d_cloud = self.pc2_to_o3d_color(msg)
            finished_time = time.perf_counter()

            self.get_logger().info(
                f"Time taken to convert pointcloud: {finished_time - start_time}"
            )

            self.get_logger().info("Processing Pointcloud... ")
            scene_pcd = self.preprocess_pointcloud(
                o3d_cloud, voxel_size=self.voxel_size
            )
            # visualize_point_cloud(scene_pcd)
            # self.visualize_point_clouds(target=scene_pcd, target_filename="handrail_offset_right.pcd")

            # TODO: get handrail pose from gz bridge, then use it as initial transformation

            result = align_pc(self.model_pcd, scene_pcd)

            if result is None:
                self.get_logger().info("ICP did not converge")
                return

            T_camera_object = np.linalg.inv(result.T_target_source)

            # This is the object's pose in camera coordinates
            position = T_camera_object[:3, 3]  # Translation vector
            rotation = T_camera_object[:3, :3]  # Rotation matrix

            predicted_image_pose = draw_pose_axes(
                self.cv_image, rotation, position, self.K
            )

            # Display the image with the pose
            cv2.imshow("Pose Estimation", predicted_image_pose)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Error processing point cloud: {e}")

    # TODO: display the estimated pose in the image
    def preprocess_pointcloud(self, o3d_msg, voxel_size=0.01):
        """
        Preprocess the pointcloud from ROS topic

        Args:
            pointcloud_msg: ROS PointCloud2 message
            voxel_size: Downsampling voxel size

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
        mask = (points_down[:, 2] > -0.7) & (points_down[:, 0] < 1)

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

        # self.visualize_point_clouds(source=self.model_pcd, target=scene_pcd_down)
        return o3d_msg

    # Example usage
    # diagnose_point_cloud(o3d_msg)

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
        self.handrail_pose = msg.poses[1] # the handrail is the second object in the list

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

        # Convert NumPy -> Open3D
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        cloud.voxel_down_sample(voxel_size=0.01)
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
