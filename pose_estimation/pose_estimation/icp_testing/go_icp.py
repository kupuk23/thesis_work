import numpy as np
import open3d as o3d
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import time
import copy
from pose_estimation.tools.pose_estimation_tools import preprocess_model

# preprocess_model(
#     # "/home/tafarrel/blender_files/grapple_fixture/grapple_fixture.obj",
#     # "/home/tafarrel/blender_files/grapple_fixture/grapple_fixture_v2.obj",
#     "/home/tafarrel/blender_files/astrobee_dock/astrobee_dock.obj",
#     file_name="astrobee_dock_ds",
#     voxel_size=0.005,
# )


def normalize_point_cloud(data_points, model_points):
    """
    Normalize point cloud to fit in [-1,1]^3 cube
    Args:
        data_points: points of source point cloud
        model_points: points of target point cloud

    Returns:
        data_normalized: Open3D point cloud object (o3d.geometry.PointCloud)
        data_centroid: centroid of the source point cloud (3x1 numpy array)
        model_normalized: Open3D point cloud object (o3d.geometry.PointCloud)
        model_centroid: centroid of the target point cloud (3x1 numpy array)
        max_scale: maximum distance from the origin (float)
    """

    # Compute centroid
    data_centroid = np.mean(data_points, axis=0)
    model_centroid = np.mean(model_points, axis=0)

    # Center the point cloud
    data_centered = data_points - data_centroid
    model_centered = model_points - model_centroid

    # Find the maximum distance from the origin
    source_max_dist = np.max(np.linalg.norm(data_centered, axis=1))
    target_max_dist = np.max(np.linalg.norm(model_centered, axis=1))
    max_scale = max(source_max_dist, target_max_dist)

    # Scale the point cloud
    data_normalized = o3d.geometry.PointCloud()
    data_normalized.points = o3d.utility.Vector3dVector(data_centered / max_scale)
    model_normalized = o3d.geometry.PointCloud()
    model_normalized.points = o3d.utility.Vector3dVector(model_centered / max_scale)

    scale = 1 / max_scale

    return (
        data_normalized,
        data_centroid,
        model_normalized,
        model_centroid,
        scale,
    )


def denormalize_transformation(transform, data_centroid, model_centroid, scale):
    """
    Convert transformation from normalized space back to original space
    """

    # Create matrices for each step
    T_data = np.eye(4)  # Translation to data centroid / source
    T_data[:3, 3] = data_centroid

    T_model = np.eye(4)  # Translation from model centroid
    T_model[:3, 3] = -model_centroid

    S = np.eye(4)  # Scaling
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    S_inv = np.eye(4)  # Inverse scaling
    S_inv[0, 0] = S_inv[1, 1] = S_inv[2, 2] = 1.0 / scale

    # Combine transformations: T_data * S_inv * transform * S * T_t
    denorm_transform = T_data @ S_inv @ transform @ S @ T_model

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
        # check if the pointcloud len is > 2000
        # if so, downsample to 2000 points
        if len(pcd.points) > 2000:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # print(f"Downsampled point cloud size: {len(pcd.points)}")

    return pcd


def init_GO_ICP():
    """
    Initialize Go-ICP
    Returns:
        goicp: Go-ICP object
    """
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

    # Higher values = more accurate but slower
    goicp.setDTSizeAndFactor(100, 3.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)

    return goicp


def go_icp(
    data_pcd,
    model_pcd,
    visualize_before=False,
    visualize_after=False,
):
    """
    Align source point cloud to target point cloud using Go-ICP.
    Args:
        data_pcd:  source point cloud
        model_pcd: target point cloud
        ds_source: Whether to downsample the source point cloud
        ds_target: Whether to downsample the target point cloud
        voxel_size: Voxel size for downsampling
        visualize_before: Whether to visualize point clouds before registration
        visualize_after: Whether to visualize registration result
    Returns:
        transform: 4x4 transformation matrix aligning source to target
    """
    # 1. Load point clouds

    # 2. COnvert PCD to numpy array
    data_points = np.asarray(data_pcd.points)
    model_points = np.asarray(model_pcd.points)

    print(
        f"Loading point clouds... {len(data_points)} points in data, {len(model_points)} points in model"
    )

    # random downsample if necessary
    if len(data_points) >= 3000:
        # Random sampling
        indices = np.random.choice(len(data_points), 3000, replace=False)
        data_points = data_points[indices]

    # Visualize before registration if requested
    if visualize_before:
        print("Original point clouds:")
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(data_points)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(model_points)
        source_pcd.paint_uniform_color([1, 0, 0])  # Red
        target_pcd.paint_uniform_color([0, 1, 0])  # Green


        source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([source_pcd, target_pcd, source_frame])

    # 2. Normalize both point clouds
    # print("Normalizing point clouds...")
    (data_normalized, data_centroid, model_normalized, model_centroid, max_scale) = (
        normalize_point_cloud(data_points, model_points)
    )

    # visualize normalized point clouds
    if visualize_before:
        print("Normalized point clouds:")
        source_temp = copy.deepcopy(data_normalized)
        source_temp.paint_uniform_color([1, 0, 0])
        model_normalized.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([source_temp, model_normalized])
        # print several lines of the point cloud
        # print("Normalized source point cloud sample:")
        # print(np.asarray(data_normalized.points)[:5])
        # print("Normalized target point cloud sample:")
        # print(np.asarray(model_normalized.points)[:5])

    # print(f"Normalization complete. Using scale factor: {max_scale}")

    # 3. Convert to POINT3D format for Go-ICP
    data_normalized_points = np.asarray(data_normalized.points)
    model_normalized_points = np.asarray(model_normalized.points)

    Nd = data_normalized_points.shape[0]
    Nm = model_normalized_points.shape[0]

    b_points = convert_to_point3d_list(data_normalized_points)
    a_points = convert_to_point3d_list(model_normalized_points)

    # 4. Initialize Go-ICP
    goicp = init_GO_ICP()

    # 5. Register the point clouds using Go-ICP
    goicp.loadModelAndData(Nm, a_points, Nd, b_points)

    # Distance transform parameters (trade-off between speed and accuracy)

    # print("Building Distance Transform...")
    goicp.BuildDT()

    start = time.time()
    # print("Starting registration...")
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
        norm_source = copy.deepcopy(data_normalized)
        norm_source.paint_uniform_color([1, 0, 0])
        norm_target = copy.deepcopy(model_normalized)
        norm_target.paint_uniform_color([0, 1, 0])
        norm_source.transform(norm_transform)
        o3d.visualization.draw_geometries([norm_source, norm_target])

    # print("Normalized transformation matrix:")
    # print(norm_transform)

    # 7. Denormalize the transformation to original coordinate frame
    transform = denormalize_transformation(
        norm_transform, data_centroid, model_centroid, max_scale
    )

    # print("Final transformation matrix:")
    # print(transform)

    # 8. Apply the transformation and visualize if requested
    if visualize_after:
        source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        source_transformed = o3d.geometry.PointCloud()
        source_transformed.points = o3d.utility.Vector3dVector(data_pcd.points)

        source_transformed.paint_uniform_color([1, 0, 0])  # Red
        model_pcd.paint_uniform_color([0, 1, 0])  # Green

        source_transformed.transform(transform)


        # print transformed point cloud

        o3d.visualization.draw_geometries(
            [source_transformed, model_pcd, source_frame]
        )

    return transform

if __name__ == "__main__":
    model = "docking_st"

    if model == "grapple":

        model_file = "/home/tafarrel/o3d_logs/grapple_fixture_v2.pcd"
        scene_file = "/home/tafarrel/o3d_logs/grapple_center.pcd"
        # scene_file = "/home/tafarrel/o3d_logs/grapple_trans_left.pcd"
        scene_file = "/home/tafarrel/o3d_logs/grapple_trans_right.pcd"
        scene_file = "/home/tafarrel/o3d_logs/grapple_right_side.pcd"
        # scene_file = "/home/tafarrel/o3d_logs/grapple_with_handrail.pcd"
    elif model == "docking_st":
        model_file = "/home/tafarrel/o3d_logs/astrobee_dock_ds.pcd"
        scene_file = "/home/tafarrel/o3d_logs/docking_front.pcd"
        scene_file = "/home/tafarrel/o3d_logs/docking_left.pcd"
    else:
        model_file = "/home/tafarrel/o3d_logs/handrail_pcd_down.pcd"
        scene_file = "/home/tafarrel/o3d_logs/handrail_origin.pcd"
        # scene_file = "/home/tafarrel/o3d_logs/handrail_right_2.pcd"
        # scene_file = "/home/tafarrel/o3d_logs/handrail_left.pcd"

    # Load point clouds
    model_pcd = loadPointCloud(model_file, ds=False, visualize=False)
    scene_pcd = loadPointCloud(scene_file, ds=True, visualize=False)

    # Run Go-ICP registration
    # MODEL == TARGET
    # DATA == SOURCE
    # TRANSFORMATION MATRIX = DATA -> MODEL

    transform = go_icp(
        model_pcd=model_pcd,
        data_pcd=scene_pcd,
        visualize_before=True,
        visualize_after=True,
    )

    print("Registration complete!")
    print("Transformation matrix:")
    print(transform)
