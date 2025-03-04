import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="world_cam_handrail.sdf",
        description="Name of the Gazebo world file to load",
    )

    model_arg = DeclareLaunchArgument(
        "model",
        default_value="robot_3d.urdf.xacro",
        description="Name of the URDF description to load",
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution(
        [
            pkg_urdf_path,  # Replace with your package name
            "urdf",
            "robots",
            LaunchConfiguration("model"),  # Replace with your URDF or Xacro file
        ]
    )

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_path, "launch", "world.launch.py"),
        ),
        launch_arguments={
            "world": LaunchConfiguration("world"),
        }.items(),
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
            {"use_sim_time": True},
        ],
    )

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "my_robot",
            "-topic",
            "robot_description",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.5",
            "-Y",
            "0.0",  # Initial spawn position
            "-P",
            "0"
        ],
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": Command(["xacro", " ", urdf_file_path]),
                "use_sim_time": True,
            },
        ],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
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
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/camera/depth@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ],
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

    relay_cmd_vel = Node(
        package="topic_tools",
        executable="relay",
        name="relay_cmd_vel",
        output="screen",
        arguments=["/cmd_vel", "model/my_robot/cmd_vel"], 
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(use_sim_time_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(gz_image_bridge_node)
    launchDescriptionObject.add_action(relay_camera_info_node)
    # launchDescriptionObject.add_action(relay_cmd_vel)

    return launchDescriptionObject
