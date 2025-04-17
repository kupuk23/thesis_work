import open3d as o3d
import numpy as np
from sklearn import linear_model
import copy


def preprocess_model(model_path, file_name="pcd_down", voxel_size=0.01):
    """
    Preprocess the 3D model from Blender

    Args:
        model_path: Path to the model file (.obj, .stl, etc.)
        voxel_size: Downsampling voxel size

    Returns:
        Preprocessed model point cloud
    """
    print(f"Loading model from {model_path}")

    # Load the model
    model = o3d.io.read_triangle_mesh(model_path)
    
    # load the model with color 
    # model_color = o3d.io.read_triangle_mesh(model_path, enable_post_processing=True)
    # model_color_pcd = model_color.sample_points_uniformly(number_of_points=10000)
    # model_color_pcd_down = model_color_pcd.voxel_down_sample(voxel_size)
    # o3d.io.write_point_cloud(f"/home/tafarrel/o3d_logs/{file_name}_color.pcd", model_color_pcd_down)

    # Ensure model has normals
    model.compute_vertex_normals()

    # Sample points from the model surface
    model_pcd = model.sample_points_uniformly(number_of_points=10000)

    # Downsample the point cloud
    model_pcd_down = model_pcd.voxel_down_sample(voxel_size)

    # save the preprocessed model
    o3d.io.write_point_cloud(f"/home/tafarrel/o3d_logs/{file_name}.pcd", model_pcd_down)

    print(f"Model preprocessed: {len(model_pcd_down.points)} points")
    return model_pcd_down


def extract_features(pcd, voxel_size):
    """
    Extract geometric features from point cloud

    Args:
        pcd: Point cloud
        voxel_size: Voxel size used for downsampling

    Returns:
        Tuple of (keypoints, feature descriptors)
    """

    # Compute FPFH features
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # We assume normals have already been computed in preprocessing

    # Extract FPFH features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd, pcd_fpfh


def initial_alignment(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    Perform initial alignment using feature matching and RANSAC

    Args:
        source: Source point cloud (model)
        target: Target point cloud (scene)
        source_fpfh: Source features
        target_fpfh: Target features
        voxel_size: Voxel size used for downsampling

    Returns:
        Transformation matrix for initial alignment
    """

    # Set RANSAC parameters
    distance_threshold = voxel_size * 1.5

    # Perform global registration using RANSAC
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )

    return result


def fine_registration(source, target, initial_transform, voxel_size):
    """
    Perform fine registration using ICP

    Args:
        source: Source point cloud (model)
        target: Target point cloud (scene)
        initial_transform: Initial transformation from coarse registration
        voxel_size: Voxel size used for downsampling

    Returns:
        Final transformation matrix
    """

    # Set ICP parameters
    distance_threshold = voxel_size * 0.5

    # Point-to-plane ICP for better accuracy
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    return result


def filter_pc_background(pointcloud):
    """filter bacground points from the point cloud
    Args:
        points: Point cloud points
        Returns:
            filtered points
    """

    # Estimate plane using RANSAC
    plane_model, inliers = pointcloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=500
    )

    # The plane model contains the coefficients [A, B, C, D] of the plane equation
    a, b, c, d = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    # Visualize the inliers (points on the plane)
    inlier_cloud = pointcloud.select_by_index(
        inliers, invert=True
    )  # Invert to get outliers

    # Run DBSCAN clustering
    labels = np.array(inlier_cloud.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

    # Color the point cloud by cluster
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    colors = np.zeros((len(labels), 3))
    colors[labels < 0] = [0, 0, 0]  # Black for noise points

    for i in range(max_label + 1):
        colors[labels == i] = [np.random.uniform(0.3, 1.0), 
                            np.random.uniform(0.3, 1.0), 
                            np.random.uniform(0.3, 1.0)]

    colored_pcd = copy.deepcopy(inlier_cloud)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([colored_pcd, coord_frame],
                                    window_name="DBSCAN Clustering Results",
                                    width=800,
                                    height=600)

    return inlier_cloud
