import numpy as np
import open3d as o3d
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import time
import copy

def normalize_point_cloud(pcd):
    """
    Normalize point cloud to fit in [-1,1]^3 cube
    """
    # Compute centroid
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    
    # Center the point cloud
    centered_pcd = copy.deepcopy(pcd)
    centered_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) - centroid
    )
    
    # Find the maximum distance from the origin
    points = np.asarray(centered_pcd.points)
    max_dist = np.max(np.linalg.norm(points, axis=1))
    
    # Scale the point cloud
    normalized_pcd = copy.deepcopy(centered_pcd)
    normalized_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(centered_pcd.points) / max_dist
    )
    
    return normalized_pcd, centroid, max_dist

def denormalize_transformation(transform, source_centroid, target_centroid, scale):
    """
    Convert transformation from normalized space back to original space
    """
    # Create matrices for each step
    T_s = np.eye(4)  # Translation to source centroid
    T_s[:3, 3] = -source_centroid
    
    T_t = np.eye(4)  # Translation from target centroid
    T_t[:3, 3] = target_centroid
    
    S = np.eye(4)    # Scaling
    S[0, 0] = S[1, 1] = S[2, 2] = scale
    
    S_inv = np.eye(4)  # Inverse scaling
    S_inv[0, 0] = S_inv[1, 1] = S_inv[2, 2] = 1.0 / scale
    
    # Combine transformations: T_s * S_inv * transform * S * T_t
    denorm_transform = T_t @ S_inv @ transform @ S @ T_s
    
    return denorm_transform

def convert_to_point3d_list(points):
    """
    Convert numpy array of points to list of POINT3D objects
    """
    p3dlist = []
    for x, y, z in points.tolist():
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return p3dlist

def loadPointCloud(filename, ds=False, visualize=False, voxel_size=0.02):
    """
    Load a point cloud from a file and optionally downsample it.
    Args:
        filename: Path to the point cloud file (pcd format)
        ds: Boolean indicating whether to downsample the point cloud
        visualize: Boolean indicating whether to visualize the point cloud
        voxel_size: Size of the voxel for downsampling
    Returns:
        pcd: Open3D point cloud object
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(filename)

    if visualize:
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
        
    if ds:
        # Voxel downsampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled point cloud size: {len(pcd.points)}")

    return pcd

def go_icp(source_file, target_file, ds_source=False, ds_target=True, 
           voxel_size=0.02, visualize_before=True, visualize_after=True):
    """
    Align source point cloud to target point cloud using Go-ICP.
    Args:
        source_file: Path to source point cloud file
        target_file: Path to target point cloud file
        ds_source: Whether to downsample the source point cloud
        ds_target: Whether to downsample the target point cloud
        voxel_size: Voxel size for downsampling
        visualize_before: Whether to visualize point clouds before registration
        visualize_after: Whether to visualize registration result
    Returns:
        transform: 4x4 transformation matrix aligning source to target
    """
    # 1. Load point clouds
    print("Loading point clouds...")
    source_pcd = loadPointCloud(source_file, ds=ds_source, voxel_size=voxel_size)
    target_pcd = loadPointCloud(target_file, ds=ds_target, voxel_size=voxel_size)
    
    # Visualize before registration if requested
    if visualize_before:
        print("Original point clouds:")
        source_temp = copy.deepcopy(source_pcd)
        source_temp.paint_uniform_color([1, 0, 0])  # Red
        target_pcd.paint_uniform_color([0, 1, 0])   # Green
        o3d.visualization.draw_geometries([source_temp, target_pcd])
        
    
    # 2. Normalize both point clouds
    print("Normalizing point clouds...")
    source_normalized, source_centroid, source_scale = normalize_point_cloud(source_pcd)
    target_normalized, target_centroid, target_scale = normalize_point_cloud(target_pcd)

    # visualize normalized point clouds
    if visualize_before:
        print("Normalized point clouds:")
        source_temp = copy.deepcopy(source_normalized)
        source_temp.paint_uniform_color([1, 0, 0])
        target_normalized.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([source_temp, target_normalized])
        # print several lines of the point cloud
        print("Normalized source point cloud sample:")
        print(np.asarray(source_normalized.points)[:5])
        print("Normalized target point cloud sample:")
        print(np.asarray(target_normalized.points)[:5])
    
    # Use the larger scale for both point clouds
    max_scale = max(source_scale, target_scale)
    
    # Re-normalize using the max scale
    source_normalized.points = o3d.utility.Vector3dVector(
        (np.asarray(source_pcd.points) - source_centroid) / max_scale
    )
    target_normalized.points = o3d.utility.Vector3dVector(
        (np.asarray(target_pcd.points) - target_centroid) / max_scale
    )
    
    print(f"Normalization complete. Using scale factor: {max_scale}")

    
    # 3. Convert to POINT3D format for Go-ICP
    source_points = np.asarray(source_normalized.points)
    target_points = np.asarray(target_normalized.points)
    
    Nm = source_points.shape[0]
    Nd = target_points.shape[0]
    
    a_points = convert_to_point3d_list(source_points)
    b_points = convert_to_point3d_list(target_points)
    
    # 4. Initialize Go-ICP
    goicp = GoICP()
    rNode = ROTNODE()
    tNode = TRANSNODE()

    # Set rotation search space (full rotation space)
    rNode.a = -3.1416
    rNode.b = -3.1416
    rNode.c = -3.1416
    rNode.w = 6.2832

    # Set translation search space (normalized to [-0.5, 0.5]³)
    tNode.x = -0.5
    tNode.y = -0.5
    tNode.z = -0.5
    tNode.w = 1.0

    # Set parameters
    goicp.MSEThresh = 0.0005  # Mean Square Error threshold
    goicp.trimFraction = 0.0  # Trimming fraction (0.0 = no trimming)

    if goicp.trimFraction < 0.001:
        goicp.doTrim = False

    # 5. Register the point clouds using Go-ICP
    goicp.loadModelAndData(Nm, a_points, Nd, b_points)
    
    # Distance transform parameters (trade-off between speed and accuracy)
    # Higher values = more accurate but slower
    goicp.setDTSizeAndFactor(100, 4.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)

    print("Building Distance Transform...")
    goicp.BuildDT()
    
    start = time.time()
    print("Starting registration...")
    goicp.Register()
    end = time.time()
    print(f"Registration completed in {end - start:.2f} seconds")
    
    # 6. Get the transformation results
    optR = np.array(goicp.optimalRotation())
    optT = goicp.optimalTranslation()
    optT.append(1.0)
    optT = np.array(optT)

    # Create the normalized transformation matrix
    norm_transform = np.eye(4)
    norm_transform[:3, :3] = optR
    norm_transform[:, 3] = optT
    
    # 6.5 visualize the normalized transformation matrix
    if visualize_after:
        print("Normalized transformation matrix:")
        norm_source = copy.deepcopy(source_normalized)
        norm_source.paint_uniform_color([1, 0, 0])
        norm_target = copy.deepcopy(target_normalized)
        norm_target.paint_uniform_color([0, 1, 0])
        norm_source.transform(norm_transform)
        o3d.visualization.draw_geometries([norm_source, norm_target])

    # print("Normalized transformation matrix:")
    # print(norm_transform)
    
    # 7. Denormalize the transformation to original coordinate frame
    transform = denormalize_transformation(
        norm_transform, source_centroid, target_centroid, max_scale)
    
    print("Final transformation matrix:")
    print(transform)
    
    # 8. Apply the transformation and visualize if requested
    if visualize_after:
        # print several lines of the point cloud
        print("Source point cloud sample:")
        print(np.asarray(source_pcd.points)[:5])
        print("Target point cloud sample:")
        print(np.asarray(target_pcd.points)[:5])
        source_transformed = copy.deepcopy(source_pcd)
        source_transformed.transform(transform)
        
        source_transformed.paint_uniform_color([1, 0, 0])  # Red
        target_pcd.paint_uniform_color([0, 1, 0])         # Green
        
        # print transformed point cloud
        print("Transformed source point cloud sample:")
        print(np.asarray(source_transformed.points)[:5])
        print("Transformed point cloud:")
        print(np.asarray(target_pcd.points)[:5])
        print("Registration result:")
        o3d.visualization.draw_geometries([source_transformed, target_pcd])
        
    
    return transform

if __name__ == "__main__":
    # Paths to your PCD files
    source_file = "/home/tafarrel/o3d_logs/grapple_fixture_down.pcd"
    target_file = "/home/tafarrel/o3d_logs/grapple_center.pcd"
    # target_file = "/home/tafarrel/o3d_logs/grapple_right_side.pcd"
    
    # Run Go-ICP registration
    transform = go_icp(
        source_file=source_file,
        target_file=target_file,
        ds_target=True,  # Downsample target
        voxel_size=0.02, # Voxel size for downsampling
        visualize_before=True,
        visualize_after=True
    )
    
    print("Registration complete!")
    print("Transformation matrix:")
    print(transform)