import numpy as np
import open3d as o3d


def visualize_registration(target_points, source_points, transformation_matrix = None):
    """
    Visualize point cloud registration results.

    Parameters:
    - target_points: Original target point cloud (Nx3 numpy array)
    - source_points: Original source point cloud (Nx3 numpy array)
    - transformation_matrix: 4x4 transformation matrix
    """
    # Create Open3D point cloud objects
    target_pcd = o3d.geometry.PointCloud()
    source_pcd = o3d.geometry.PointCloud()

    # Set points
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    # Color the point clouds differently
    target_pcd.paint_uniform_color([1, 0, 0])  # Red for target
    source_pcd.paint_uniform_color([0, 0, 1])  # Blue for source

    if transformation_matrix is None:
        transformation_matrix = np.eye(4)  # Identity matrix
        
    # Transform the source point cloud
    transformed_source_pcd = source_pcd.transform(transformation_matrix)

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point clouds to visualization
    vis.add_geometry(target_pcd)
    vis.add_geometry(transformed_source_pcd)

    # Set up the view
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.asarray([1, 1, 1])  # White background

    # Adjust the view
    vis.update_geometry(target_pcd)
    vis.update_geometry(transformed_source_pcd)
    vis.poll_events()
    vis.update_renderer()

    # Wait for window to be closed
    vis.run()
    vis.destroy_window()
