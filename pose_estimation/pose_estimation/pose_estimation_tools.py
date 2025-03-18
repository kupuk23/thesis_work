import open3d as o3d
import numpy as np


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

def calculate_pose_from_transformation(transformation_matrix):
    """
    Extract position and orientation from transformation matrix
    
    Args:
        transformation_matrix: 4x4 transformation matrix
        
    Returns:
        tuple (position, quaternion)
    """
    # Extract rotation matrix (3x3) and translation vector (3x1)
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    # trace of rotation matrix
    trace = np.trace(rotation_matrix)
    
    # Check the trace to determine the best computation method
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        qy = 0.25 * S
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        qz = 0.25 * S
    
    # Return position and quaternion
    position = translation_vector
    quaternion = np.array([qw, qx, qy, qz])
    
    return position, quaternion