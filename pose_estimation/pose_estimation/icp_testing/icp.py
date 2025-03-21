import small_gicp
import open3d as o3d
import numpy as np


pcd_source = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/source.pcd")
pcd_target = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/target.pcd")
pcd_target_offset = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/target_2.pcd")


def view_pc(pcd_source, pcd_target):
    # view 2 point clouds together with different color
    pcd_source.paint_uniform_color([1, 0, 0])
    pcd_target.paint_uniform_color([0, 0, 1])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([pcd_source, coordinate_frame])


def align_pc(pcd_source, pcd_target):
    source = np.asarray(pcd_source.points)  # Mx3 numpy array
    target = np.asarray(pcd_target.points)  # Nx3 numpy array

    # rotate target PC
    target = np.dot(target, np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    target = np.dot(target, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
    pcd_target.points = o3d.utility.Vector3dVector(target)

    # visualize source pointcloud
    # o3d.visualization.draw_geometries([pcd_source, pcd_target])

    result = small_gicp.align(target, source, downsampling_resolution=0.25)
    if result.converged:
        print("Small GICP converged!")
        # print(f"Final transformation matrix: {result.T_target_source}")
        # print(f"Number of iterations: {result.iterations}")
        # print(f"error: {result.error}")
        # visualize_registration(target, source, result.T_target_source)
        return result

    else:
        print("Small GICP did not converge.")
        return None


def visualize_registration(target_points, source_points, transformation_matrix):
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


# Example usage:
# Assuming you have:
# target_raw_numpy: target point cloud
# source_raw_numpy: source point cloud
# result.T_target_source: transformation matrix from small_gicp
# visualize_registration(target, source, result.T_target_source)

if __name__ == "__main__":
    view_pc(pcd_source, pcd_target)
    align_pc(pcd_source, pcd_target)
