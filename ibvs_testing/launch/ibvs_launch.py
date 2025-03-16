from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package name - replace 'my_package' with your actual package name
    package_name = "ibvs_testing"

    # Optional: Create launch arguments for parameters you want to set from the command line
    # node_name_arg = DeclareLaunchArgument(
    #     "node_name", default_value="ibvs_node", description="Name for the node"
    # )

    # Define the node to be launched
    my_node = Node(
        package=package_name,
        executable="ibvs_node",  # your script entry point
        name="ibvs_node",
        output="screen",
    )

    # Return the launch description
    return LaunchDescription([my_node])
