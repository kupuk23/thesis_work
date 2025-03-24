import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
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

    spawn_robot_arg = DeclareLaunchArgument(
        "spawn_robot",
        default_value="default",
        description="Spawn the robot in the Gazebo world",
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

    def create_robot_spawner(name, x, y, z, R, P, Y, namespace=None):
        """Create a node for spawning a robot"""
        args = [
            "-name",
            name,
            "-topic",
            "robot_description",
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
            "-R",
            str(R),
            "-P",
            str(P),
            "-Y",
            str(Y),
        ]

        if namespace:
            args.extend(["-namespace", namespace])

        return Node(
            package="ros_gz_sim",
            executable="create",
            arguments=args,
            output="screen",
            parameters=[{"use_sim_time": True}],
        )

    # def spawn_robot_pose(scenario : LaunchConfiguration):
    #     if scenario == "default":
    #         return create_robot_spawner("my_robot", -2, 0, 1.25, 0.0, 0, 0)
    #     elif scenario == "ibvs_sample":
    #         return create_robot_spawner("my_robot", -2, 0.8, 1.4, 0.6, -0.3, -0.4)
    #     elif scenario == "handrail":
    #         return create_robot_spawner("my_robot", -2, 0, 1.25, 0.0, 0, 0)
    #     else:
    #         print(f"Unknown scenario: {scenario}")
    #         return create_robot_spawner("my_robot", 0, 0, 1.25, 0.0, 0, 0)

    # Spawn the URDF model using the `/world/<world_name>/create` service

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
            # add the world pose
            "/world/iss_world/pose/info@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V"
            # "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/world/iss_world/control@ros_gz_interfaces/srv/ControlWorld",
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
    ibvs_node = Node(
        package="ibvs_testing",
        executable="ibvs_node",
        name="ibvs_node",
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
    launchDescriptionObject.add_action(spawn_robot_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(rviz_node)
    # launchDescriptionObject.add_action(create_robot_spawner("my_robot", -2, 0, 1.25, 0.0, 0, 0)) # default case
    # launchDescriptionObject.add_action(create_robot_spawner("my_robot", -2, 0.8, 1.4, 0.6, -0.3, -0.4)) # ibvs_sample case
    # launchDescriptionObject.add_action(create_robot_spawner("my_robot", -1.26, 0, 1.44, 0, -0.04, 0)) # PnP case
    launchDescriptionObject.add_action(create_robot_spawner("my_robot", -2, 3, 1, 0.0, 0, 0)) # handrail case
    launchDescriptionObject.add_action(robot_state_publisher_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(gz_image_bridge_node)
    launchDescriptionObject.add_action(relay_camera_info_node)
    # launchDescriptionObject.add_action(ibvs_node)
    # launchDescriptionObject.add_action(depth_image_proc_node)
    # launchDescriptionObject.add_action(relay_cmd_vel)

    return launchDescriptionObject
