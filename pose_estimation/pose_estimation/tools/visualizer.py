import open3d as o3d
import numpy as np
import copy

def visualize_point_cloud(pcd, window_name="RGB Point Cloud Visualization"):
    """
    Visualize a point cloud with its RGB colors
    
    Parameters:
        pcd: open3d.geometry.PointCloud - the point cloud to visualize
        window_name: str - title for the visualization window
    """
    if len(pcd.points) == 0:
        print("Error: Cannot visualize empty point cloud")
        return
    
    # Print basic information about the point cloud
    print(f"Point cloud contains {len(pcd.points)} points")
    print(f"Has colors: {len(pcd.colors) > 0}")
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Get render option
    render_option = vis.get_render_option()
    
    # Improve visualization settings
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    render_option.point_size = 2.0  # Larger points
    render_option.show_coordinate_frame = True  # Show coordinate frame
    
    # Optimize view
    vis.get_view_control().set_zoom(0.8)
    
    # Update visualization
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # Run the visualization - this will block until window is closed
    vis.run()
    vis.destroy_window()

def enhanced_visualization(pcd, save_image=False, image_path="pointcloud.png"):
    """
    Enhanced point cloud visualization with more features
    
    Parameters:
        pcd: open3d.geometry.PointCloud - the point cloud to visualize
        save_image: bool - whether to save a screenshot
        image_path: str - path to save the screenshot
    """
    # Create a copy to avoid modifying the original
    pcd_copy = copy.deepcopy(pcd)
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Add the point cloud
    vis.add_geometry(pcd_copy)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Darker background
    render_option.point_size = 2.0
    render_option.show_coordinate_frame = True
    
    # Add a coordinate frame for scale reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Setup view
    view_control = vis.get_view_control()
    
    # Set a default viewpoint
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.8)
    
    # Update visualization
    vis.update_geometry(pcd_copy)
    vis.poll_events()
    vis.update_renderer()
    
    # Save image if requested
    if save_image:
        vis.capture_screen_image(image_path)
        print(f"Image saved to {image_path}")
    
    print("Point Cloud Viewer Controls:")
    print("  Left-click + drag: Rotate")
    print("  Ctrl + Left-click + drag: Pan")
    print("  Scroll wheel: Zoom in/out")
    print("  '[' or ']': Decrease/Increase point size")
    print("  'R': Reset view")
    print("  'Q', 'Esc', or closing the window: Exit")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def rotating_view(pcd, save_video=False, video_path="pointcloud_video.mp4"):
    """
    Create a rotating view of the point cloud
    
    Parameters:
        pcd: open3d.geometry.PointCloud - the point cloud to visualize
        save_video: bool - whether to save frames for a video
        video_path: str - base path for video frames
    """
    # Create a custom visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Add the point cloud
    vis.add_geometry(pcd)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    render_option.point_size = 2.0
    
    # Get the view control
    view_control = vis.get_view_control()
    
    # Setup camera
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.7)
    
    # Rotate the view
    print("Creating rotating view. Press 'Q' to exit.")
    for i in range(360):
        # Rotate by 1 degree each step
        vis.poll_events()
        vis.update_renderer()
        
        # Rotate camera around the z-axis
        view_control.rotate(1.0, 0.0)  # Rotate 1 degree horizontally
        
        # Save frame if requested
        if save_video:
            filename = f"{video_path}_{i:04d}.png"
            vis.capture_screen_image(filename)
        
        # Exit if 'q' is pressed
        if vis.poll_events():
            if ord('q') in vis.get_key_pressed():
                break
    
    vis.destroy_window()
    
    if save_video:
        print(f"Saved {360} frames with prefix {video_path}_")
        print("You can convert them to a video using ffmpeg:")
        print(f"ffmpeg -framerate 30 -i {video_path}_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

# Example usage:
# Basic visualization
# visualize_point_cloud(scene_pcd)

# Enhanced visualization with option to save screenshot
# enhanced_visualization(scene_pcd, save_image=True, image_path="my_pointcloud.png")

# Create a rotating view (with optional video saving)
# rotating_view(scene_pcd, save_video=False)