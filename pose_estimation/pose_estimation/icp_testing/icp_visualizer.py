import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

def visualize_pose_in_image(source_pcd_path, target_pcd_path, transformation_matrix, 
                           camera_intrinsic, image_path):
    """
    Visualize the aligned object frame in a 2D camera image.
    
    Args:
        source_pcd_path: Path to the source point cloud file (.pcd)
        target_pcd_path: Path to the target point cloud file (.pcd)
        transformation_matrix: 4x4 transformation matrix from target to source
        camera_intrinsic: 3x3 camera intrinsic matrix
        image_path: Path to the camera image
    """
    # Load point clouds
    source_pcd = o3d.io.read_point_cloud(source_pcd_path)
    target_pcd = o3d.io.read_point_cloud(target_pcd_path)
    
    # Apply transformation to align target with source
    target_aligned = target_pcd.transform(transformation_matrix)
    
    # Load camera image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create coordinate frame for visualization (origin and axes)
    # Get the centroid of the aligned target as the origin
    target_points = np.asarray(target_aligned.points)
    origin = np.mean(target_points, axis=0)
    
    # Define axes with appropriate length (adjust scale as needed)
    axis_length = 0.1  # in meters, adjust based on your point cloud scale
    axes_points = np.array([
        origin,  # Origin
        origin + np.array([axis_length, 0, 0]),  # X-axis (red)
        origin + np.array([0, axis_length, 0]),  # Y-axis (green)
        origin + np.array([0, 0, axis_length])   # Z-axis (blue)
    ])
    
    # Project 3D points to 2D image space
    points_2d = project_points_to_image(axes_points, camera_intrinsic)
    
    # Draw the coordinate frame on the image
    draw_coordinate_frame(img, points_2d)
    
    # Optionally, project and visualize point clouds as well
    source_points_2d = project_points_to_image(np.asarray(source_pcd.points), camera_intrinsic)
    target_points_2d = project_points_to_image(np.asarray(target_aligned.points), camera_intrinsic)
    
    draw_point_cloud(img, source_points_2d, color=(255, 0, 0), alpha=0.5)  # Red for source
    draw_point_cloud(img, target_points_2d, color=(0, 0, 255), alpha=0.5)  # Blue for target
    
    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title("Object Pose Visualization")
    plt.axis('off')
    plt.show()
    
    # Evaluate ICP accuracy
    evaluate_icp_accuracy(source_pcd, target_pcd, target_aligned, transformation_matrix)
    
    return img

def project_points_to_image(points_3d, camera_intrinsic):
    """
    Project 3D points to 2D image space using the camera intrinsic matrix.
    Assumes the points are already in the camera coordinate system.
    """
    points_2d = []
    for point in points_3d:
        # Convert to homogeneous coordinates
        x, y, z = point
        
        # Skip points behind the camera
        if z <= 0:
            continue
            
        # Project
        u = (camera_intrinsic[0, 0] * x / z) + camera_intrinsic[0, 2]
        v = (camera_intrinsic[1, 1] * y / z) + camera_intrinsic[1, 2]
        
        points_2d.append([int(u), int(v)])
    
    return np.array(points_2d)

def draw_coordinate_frame(image, points_2d):
    """
    Draw the coordinate frame on the image.
    points_2d should contain origin and the three axis endpoints.
    """
    if len(points_2d) < 4:
        print("Not enough points to draw coordinate frame")
        return
    
    origin = tuple(points_2d[0])
    x_axis = tuple(points_2d[1])
    y_axis = tuple(points_2d[2])
    z_axis = tuple(points_2d[3])
    
    # Draw axes with different colors
    cv2.line(image, origin, x_axis, (255, 0, 0), 2)  # X-axis: Red
    cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis: Green
    cv2.line(image, origin, z_axis, (0, 0, 255), 2)  # Z-axis: Blue
    
    # Label the axes
    cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_point_cloud(image, points_2d, color=(0, 255, 0), alpha=0.5):
    """
    Draw the projected point cloud on the image.
    """
    overlay = image.copy()
    
    for point in points_2d:
        cv2.circle(overlay, tuple(point), 1, color, -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def evaluate_icp_accuracy(source_pcd, target_pcd, target_aligned, transformation_matrix):
    """
    Evaluate the accuracy of the ICP alignment.
    """
    # Calculate the RMSE between source and aligned target
    source_points = np.asarray(source_pcd.points)
    aligned_points = np.asarray(target_aligned.points)
    
    # Find the nearest neighbor of each source point in the aligned target
    distances = []
    for point in source_points:
        min_dist = np.min(np.linalg.norm(aligned_points - point, axis=1))
        distances.append(min_dist)
    
    rmse = np.sqrt(np.mean(np.array(distances) ** 2))
    
    # Extract rotation and translation from transformation matrix
    rotation = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]
    
    # Convert rotation to Euler angles (in degrees)
    rotation_euler = rotation_matrix_to_euler_angles(rotation) * 180 / np.pi
    
    print("ICP Accuracy Evaluation:")
    print(f"RMSE: {rmse:.4f} units")
    print(f"Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
    print(f"Rotation (Euler angles in degrees): [{rotation_euler[0]:.2f}, {rotation_euler[1]:.2f}, {rotation_euler[2]:.2f}]")

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    """
    # Check if the rotation matrix is valid
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
        
    return np.array([x, y, z])

# Example usage:
"""
# Load your data
source_pcd_path = "source.pcd"
target_pcd_path = "target.pcd"
image_path = "camera_view.jpg"

# Define camera intrinsic matrix
camera_intrinsic = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Define the transformation matrix (from ICP result)
transformation_matrix = np.eye(4)  # Replace with your ICP result

# Visualize the pose in the image
result_image = visualize_pose_in_image(
    source_pcd_path, 
    target_pcd_path, 
    transformation_matrix, 
    camera_intrinsic, 
    image_path
)
"""