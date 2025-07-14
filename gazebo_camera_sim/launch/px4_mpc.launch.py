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

    pkg_urdf_path = get_package_share_directory("camera_description")
    pkg_gazebo_path = get_package_share_directory("gazebo_camera_sim")

    gazebo_models_path, ignore_last_dir = os.path.split(pkg_urdf_path)
    # os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    rviz_launch_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Open RViz."
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time.",
    )

    # Launch rviz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "-d",
            os.path.join(pkg_urdf_path, "rviz", "camera_description.rviz"),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )




    # Node to bridge messages like /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",  # [ means the message type is from gazebo
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",  # @ means the message type is bi directional
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/world/default/dynamic_pose/info@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V",
            # "/world/iss_world/pose/info@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V",
            # "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/world/iss_world/control@ros_gz_interfaces/srv/ControlWorld",
        ],
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        # remappings=[("/tf", "tf_gz")],  # Remap tf to tf_gz
    )

    # Node to bridge camera image with image_transport and compressed_image_transport
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=[
            "/camera/image",
        ],
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "camera.image.compressed.jpeg_quality": 75,
            },
        ],
    )

    tf_camera_link_pub = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='spacecraft_to_camera_tf',
    arguments=['-0.09', '0.0', '0.58', '3.14', '0.0', '0.0', 'robot_base_imu', 'camera_link']
)
    # add perception pipeline with config file
    
    perception_pkg_dir = get_package_share_directory('pose_estimation_pcl')
    # Path to the config file
    perception_config_file = os.path.join(perception_pkg_dir, 'config', 'pose_estimation_config.yaml')

    pose_estimation_node = Node(
        package='pose_estimation_pcl',
        executable='pose_estimation_pcl',
        name='pose_estimation_pcl',
        parameters=[perception_config_file],
        # prefix='gdbserver localhost:3000',
        output='screen',
    )

    ibvs_node = Node(
        package="ibvs_testing",
        executable="ibvs_node",
        name="ibvs_node",
        output="screen",
    )

    # add tf_broadcaster
    tf_broadcaster = Node(
        package="tf_handler",
        executable="gz_pose_transform",
        name="gz_pose_transform",
        output="screen",
    )

    # Relay node to republish /camera/camera_info to /camera/image/camera_info
    relay_camera_info_node = Node(
        package="topic_tools",
        executable="relay",
        name="relay_camera_info",
        output="screen",
        arguments=["camera/camera_info", "camera/image/camera_info"],
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(use_sim_time_arg)
    # launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(gz_image_bridge_node)
    launchDescriptionObject.add_action(relay_camera_info_node)
    # launchDescriptionObject.add_action(tf_broadcaster)
    launchDescriptionObject.add_action(pose_estimation_node)
    launchDescriptionObject.add_action(tf_camera_link_pub)
    # launchDescriptionObject.add_action(tf_world_publisher)

    return launchDescriptionObject
