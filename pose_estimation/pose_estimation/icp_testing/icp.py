import small_gicp
import open3d as o3d
import numpy as np
import cv2
from pose_estimation.tools.pose_estimation_tools import preprocess_model
from pose_estimation.icp_testing.icp_visualizer import visualize_pose_in_image


img_target_left = cv2.imread("/home/tafarrel/handrail_test2.jpg")
img_target_right = cv2.imread("/home/tafarrel/handrail_offset_right.jpg")
img_source = cv2.imread("/home/tafarrel/handrail.jpg")
K = np.array(
                [
                    [500.0, 0.0, 320.0],  # fx, 0, cx
                    [0.0, 500.0, 240.0],  # 0, fy, cy
                    [0.0, 0.0, 1.0],  # 0, 0, 1
                ]
            )

test = 1 # 1 for center, 2 for left, 3 for right
# preprocess_model(
#             "/home/tafarrel/blender_files/handrail/handrail.obj",
#             voxel_size=0.005,
#         )

pcd_source = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/handrail_pcd_down.pcd")
pcd_target = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/handrail_origin.pcd")
pcd_target_offset_right = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/handrail_offset_right.pcd")
pcd_target_offset_left = o3d.io.read_point_cloud("/home/tafarrel/o3d_logs/handrail_test2.pcd")



def view_pc(pcd_source, pcd_target):
    # view 2 point clouds together with different color
    
    # pcd_source.paint_uniform_color([1, 0, 0])
    # pcd_target.paint_uniform_color([0, 0, 1])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([pcd_source, pcd_target, coordinate_frame])


def align_pc(pcd_source, pcd_target, init_T=None):
    source = np.asarray(pcd_source.points)  # Mx3 numpy array
    target = np.asarray(pcd_target.points)  # Nx3 numpy array

    # perform inittial alignment to target to visualize
    # pcd_source.points = o3d.utility.Vector3dVector(source)
    # pcd_target.points = o3d.utility.Vector3dVector(target)

    

    # view_pc(pcd_source, pcd_target)

    # visualize source pointcloud
    # o3d.visualization.draw_geometries([pcd_source, pcd_target])

    result = small_gicp.align(target, source, registration_type='VGICP' , downsampling_resolution=0.1, init_T_target_source=init_T)
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

def draw_pose_axes(image, rotation, translation, camera_matrix, dist_coeffs=None, axis_length=0.2):
    """
    Draw 3D coordinate axes on the image to visualize the estimated pose.
    
    Args:
        image: Target image to draw on
        rotation: Rotation vector or matrix from ICP
        translation: Translation vector from ICP
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients (optional)
        axis_length: Length of the axes to draw (in meters)
    
    Returns:
        Image with drawn coordinate axes
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    # Convert rotation to rotation vector if it's a matrix
    if rotation.shape[0] == 3 and rotation.shape[1] == 3:
        rotation, _ = cv2.Rodrigues(rotation)
    
    # Define coordinate axes in 3D space
    axis_points = np.float32([[0, 0, 0],
                              [axis_length, 0, 0],  # X-axis (Red)
                              [0, axis_length, 0],  # Y-axis (Green)
                              [0, 0, axis_length]]) # Z-axis (Blue)
    
    # Project 3D points to the image plane
    image_points, _ = cv2.projectPoints(axis_points, rotation, translation, camera_matrix, dist_coeffs)
    image_points = np.int32(image_points).reshape(-1, 2)
    
    # Draw the axes
    origin = tuple(image_points[0])
    image = cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X-axis in Red
    image = cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y-axis in Green
    image = cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z-axis in Blue
    
    return image


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
    if test == 1:
        
        # view_pc(pcd_source, pcd_target)
        T_matrix = align_pc(pcd_source, pcd_target)
        translation = T_matrix.T_target_source[:3, 3]
        rotation = T_matrix.T_target_source[:3, :3]

        print(translation)
        print(rotation)
        
    
        image = visualize_pose_in_image(img_source, T_matrix.T_target_source, K)
        # image = draw_pose_axes(img_source, rotation, translation, K, axis_length=0.1)
    # image = draw_pose_axes(img_source, rotation, translation, K, axis_length=0.1)
    elif test == 2:
        T_matrix = align_pc(pcd_source, pcd_target_offset_left)
        translation = T_matrix.T_target_source[:3, 3]
        rotation = T_matrix.T_target_source[:3, :3]

        print(translation)
        print(rotation)       
        image = draw_pose_axes(img_target_left, rotation, translation, K, axis_length=0.1)
        # image = visualize_object_frame_in_image(img_target_left, T_matrix.T_target_source, K)
    elif test == 3:
        T_matrix = align_pc(pcd_source, pcd_target_offset_right)
        translation = T_matrix.T_target_source[:3, 3]
        rotation = T_matrix.T_target_source[:3, :3]

        print(translation)
        print(rotation)       
        image = draw_pose_axes(img_target_right, rotation, translation, K, axis_length=0.1)
        # image = visualize_object_frame_in_image(img_target_right, T_matrix.T_target_source, K)
    cv2.imshow("image", image)
    cv2.waitKey(0)