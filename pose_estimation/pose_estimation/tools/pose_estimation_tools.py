import open3d as o3d
import numpy as np

def preprocess_model(model_path, voxel_size=0.01):
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

        print(
            f"Model preprocessed: {len(model_pcd_down.points)} points"
        )
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
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
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
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
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
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    
    
    return result
