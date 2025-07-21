import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)

from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    perception_pkg_dir = get_package_share_directory("pose_estimation_pcl")
    # Path to the config file
    perception_config_file = os.path.join(
        perception_pkg_dir, "config", "pose_estimation_config.yaml"
    )

    tf_camera_link_pub = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="spacecraft_to_camera_tf",
        arguments=[
            "-0.09",
            "0.0",
            "0.58",
            "3.14",
            "0.0",
            "0.0",
            "map", #robot_base_imu
            "zed_camera_link",
        ],
    )

    pose_estimation_node = Node(
        package="pose_estimation_pcl",
        executable="pose_estimation_pcl",
        name="pose_estimation_pcl",
        parameters=[perception_config_file],
        # prefix='gdbserver localhost:3000',
        # arguments=["--ros-args", "--log-level", "debug"],
        output="screen",
    )

    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(pose_estimation_node)
    launchDescriptionObject.add_action(tf_camera_link_pub)

    return launchDescriptionObject