from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('pose_estimation_pcl')
    
    # Path to the config file
    config_file = os.path.join(package_dir, 'config', 'pose_estimation_config.yaml')
    
    # Create the node with parameters
    pose_estimation_node = Node(
        package='pose_estimation_pcl',
        executable='pose_estimation_pcl',
        name='pose_estimation_pcl',
        parameters=[config_file],
        # prefix='gdbserver localhost:3000',
        output='screen',
    )
    
    # Return the launch description
    return LaunchDescription([
        pose_estimation_node
    ])