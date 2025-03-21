import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import copy
import cv2
from cv_bridge import CvBridge
import struct
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import sensor_msgs_py.point_cloud2 as pc2
# from icp_testing.icp import align_pc
from icp_testing.pnp_tracker import PnPTracker


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        self.voxel_size = 0.01

        # Load the model (only needs to be done once)
        # self.model_pcd = self.preprocess_model(
        #     "/home/tafarrel/blender_files/handrail/handrail.obj",
        #     voxel_size=self.voxel_size,
        # )

        self.model_pcd = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd")
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
            PointCloud2, "/camera/points", self.pointcloud_callback, 10
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
        o3d.io.write_point_cloud("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd", model_pcd_down)

        self.get_logger().info(
            f"Model preprocessed: {len(model_pcd_down.points)} points"
        )
        return model_pcd_down

    def pointcloud_callback(self, msg):
        """Process incoming pointcloud data"""
        self.latest_pointcloud = msg

        # Process the pointcloud directly
        # scene_pcd = self.preprocess_pointcloud(msg, voxel_size=self.voxel_size)
        # result = align_pc(self.model_pcd, scene_pcd)

        # T_camera_object = np.linalg.inv(result.T_target_source)

        # # This is the object's pose in camera coordinates
        # position = T_camera_object[:3, 3]  # Translation vector
        # rotation = T_camera_object[:3, :3]  # Rotation matrix

        # print(f"Object position in camera frame: {position} and rotation: {rotation}")

        # Now you can use this for pose estimation
        # if scene_pcd is not None:
        #     self.process_for_pose_estimation(scene_pcd)

    def preprocess_pointcloud(self, pointcloud_msg, voxel_size=0.01):
        """
        Preprocess the pointcloud from ROS topic

        Args:
            pointcloud_msg: ROS PointCloud2 message
            voxel_size: Downsampling voxel size

        Returns:
            Preprocessed scene point cloud as Open3D point cloud
        """

        # Convert ROS -> NumPy
        points_list = []
        colors_list = []
        for point in pc2.read_points(
            pointcloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True
        ):

            points_list.append([point[0], point[1], point[2]])
            packed_rgb = struct.unpack("I", struct.pack("f", point[3]))[0]

            # Extract individual color components
            r = np.bitwise_and(np.right_shift(packed_rgb, 16), 255).astype(np.uint8)
            g = np.bitwise_and(np.right_shift(packed_rgb, 8), 255).astype(np.uint8)
            b = np.bitwise_and(packed_rgb, 255).astype(np.uint8)

            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

        # Create Open3D point cloud
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points_list)
        # Add the color information to the point cloud
        scene_pcd.colors = o3d.utility.Vector3dVector(colors_list)

        # Remove statistical outliers
        scene_pcd, _ = scene_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        # Downsample the point cloud
        scene_pcd_down = scene_pcd.voxel_down_sample(voxel_size)

        # filter maximum depth by z
        points_down = np.asarray(scene_pcd_down.points)
        colors_down = np.asarray(scene_pcd_down.colors)
        mask = points_down[:, 2] > -0.7
        filtered_points = points_down[mask]
        filtered_colors = colors_down[mask]

        scene_pcd_down.points = o3d.utility.Vector3dVector(filtered_points)
        scene_pcd_down.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Estimate normals
        # scene_pcd_down.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #         radius=voxel_size * 2, max_nn=30
        #     )
        # )

        self.get_logger().info(
            f"Pointcloud processed: {len(scene_pcd_down.points)} points"
        )
        # self.visualize_point_cloud_matplotlib(scene_pcd_down)
        # Add debugging - check the bounds and orientation
        points_array = np.array(scene_pcd_down.points)
        min_bounds = np.min(points_array, axis=0)
        max_bounds = np.max(points_array, axis=0)
        center = np.mean(points_array, axis=0)
        
        self.get_logger().info(f"Point cloud bounds: min={min_bounds}, max={max_bounds}")
        self.get_logger().info(f"Point cloud center: {center}")


        # self.visualize_point_clouds(source=self.model_pcd, target=scene_pcd_down)
        return scene_pcd_down

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

    def rgb_callback(self, msg):
        """Process incoming RGB image"""
        try:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            tracker.run_program(self.cv_image)
            # cv2.imwrite("/home/tafarrel/handrail.jpg", cv_image)
        except:
            self.get_logger().info("Error converting image")
        # Could be used for visualization or feature extraction from RGB

    def visualize_point_clouds(self, source, target, transformed_source=None):
        """
        Visualize the point clouds (non-blocking in separate process)

        Args:
            source: Source point cloud (model)
            target: Target point cloud (scene)
            transformed_source: Transformed source after registration (if available)
        """
        # Create visualization geometries
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # Color the point clouds
        source_temp.paint_uniform_color([1, 0, 0])  # Red for model
        target_temp.paint_uniform_color([0, 1, 0])  # Green for scene

        geometries = [source_temp, target_temp]

        # Add transformed source if available
        if transformed_source is not None:
            transformed_temp = copy.deepcopy(transformed_source)
            transformed_temp.paint_uniform_color(
                [0, 0, 1]
            )  # Blue for transformed model
            geometries.append(transformed_temp)

        # Save to file for later visualization (non-blocking)
        o3d.io.write_point_cloud("/home/tafarrel/o3d_logs/source.pcd", source_temp)
        o3d.io.write_point_cloud("/home/tafarrel/o3d_logs/target.pcd", target_temp)
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
