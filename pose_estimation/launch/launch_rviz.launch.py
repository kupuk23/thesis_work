import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_urdf_path = get_package_share_directory("pose_estimation")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            os.path.join(pkg_urdf_path, "rviz", "rviz.rviz"),
        ],
        parameters=[
            {"use_sim_time": True},
        ],
    )

    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(rviz_node)

    return launchDescriptionObject
