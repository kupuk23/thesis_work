import numpy as np
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped
import tf_transformations

def pose_to_matrix(pose_msg):
    """
    Convert a PoseStamped message to a 4x4 transformation matrix.
    
    Args:
        pose_msg (PoseStamped): The ROS PoseStamped message
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    # Extract position
    position = pose_msg.position
    position_vector = np.array([position.x, position.y, position.z])
    
    # Extract orientation as quaternion
    orientation = pose_msg.orientation
    quaternion = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
    
    # Create rotation matrix from quaternion
    rotation_matrix = tf_transformations.quaternion_matrix(quaternion)
    
    # Set translation in transformation matrix
    rotation_matrix[0:3, 3] = position_vector
    
    return rotation_matrix

from geometry_msgs.msg import Transform, Pose

def transform_to_pose(transform):
    """
    Convert a Transform message to a Pose message
    
    Args:
        transform (geometry_msgs.msg.Transform): The transform to convert
        
    Returns:
        geometry_msgs.msg.Pose: The resulting pose
    """
    pose = Pose()
    
    # Copy the translation
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    
    # Copy the rotation
    pose.orientation.x = transform.rotation.x
    pose.orientation.y = transform.rotation.y
    pose.orientation.z = transform.rotation.z
    pose.orientation.w = transform.rotation.w
    
    return pose

def transform_to_matrix(transform_msg):
    """
    Convert a Transform or TransformStamped message to a 4x4 transformation matrix.
    
    Args:
        transform_msg: Either a Transform or TransformStamped ROS message
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    # Handle both Transform and TransformStamped messages
    if isinstance(transform_msg, TransformStamped):
        transform = transform_msg.transform
    else:
        transform = transform_msg
    
    # Extract translation
    translation = transform.translation
    translation_vector = np.array([translation.x, translation.y, translation.z])
    
    # Extract rotation as quaternion
    rotation = transform.rotation
    quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    
    # Create rotation matrix from quaternion
    matrix = tf_transformations.quaternion_matrix(quaternion)
    
    # Set translation in transformation matrix
    matrix[0:3, 3] = translation_vector
    
    return matrix

def matrix_to_transform(matrix):
    """
    Convert a 4x4 transformation matrix to a Transform message.
    
    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix
        
    Returns:
        geometry_msgs.msg.Transform: ROS Transform message
    """
    transform = Transform()
    
    # Extract translation
    transform.translation.x = matrix[0, 3]
    transform.translation.y = matrix[1, 3]
    transform.translation.z = matrix[2, 3]
    
    # Extract rotation and convert to quaternion
    quaternion = tf_transformations.quaternion_from_matrix(matrix)
    transform.rotation.x = quaternion[0]
    transform.rotation.y = quaternion[1]
    transform.rotation.z = quaternion[2]
    transform.rotation.w = quaternion[3]
    
    return transform

def matrix_to_pose(matrix):
    """
    Convert a 4x4 transformation matrix to a Pose.
    
    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix
        
    Returns:
        geometry_msgs.msg.Pose: ROS Pose message
    """
    from geometry_msgs.msg import Pose
    
    pose = Pose()
    
    # Extract translation
    pose.position.x = matrix[0, 3]
    pose.position.y = matrix[1, 3]
    pose.position.z = matrix[2, 3]
    
    # Extract rotation and convert to quaternion
    quaternion = tf_transformations.quaternion_from_matrix(matrix)
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    
    return pose

def get_camera_to_object_transform(world_to_object, world_to_camera):
    """
    Calculate the transformation from camera to object
    
    Args:
        world_to_object (numpy.ndarray): 4x4 transformation matrix from world to object
        world_to_camera (numpy.ndarray): 4x4 transformation matrix from world to camera
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from camera to object
    """
    # Calculate camera to world (inverse of world to camera)
    camera_to_world = np.linalg.inv(world_to_camera)
    
    # Calculate camera to object: T_camera_object = T_camera_world * T_world_object
    camera_to_object = np.matmul(camera_to_world, world_to_object)
    
    return camera_to_object