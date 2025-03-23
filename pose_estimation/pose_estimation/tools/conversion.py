import open3d as o3d
import numpy as np
import struct
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField


class Converter:
    def __init__(self):
        self.min_intensity = -1
        self.max_intensity = -1

    def normalize_np(self, array, alpha, beta):
        # Calculate min and max values of the array
        min_val = np.min(array)
        max_val = np.max(array)

        # Normalize the array to the range [alpha, beta]
        normalized_array = (array - min_val) / (max_val - min_val) * (beta - alpha) + alpha

        return normalized_array

    def O3DPointCloud_to_nparray(self, open3d_cloud):
        np_open3d_cloud_coordinates = np.asarray(open3d_cloud.points)
        colors = np.asarray(open3d_cloud.colors)
        return np_open3d_cloud_coordinates.reshape(-1, 3), colors

    def nparray_to_ROSpc2(self, np_open3d_cloud, frame_id="map", np_open3d_colors=None):
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        # Create a structured dtype for the point cloud with correct offsets
        point_step = 24  # Based on your message info
        
        # Prepare data array with appropriate size
        if np_open3d_colors is not None and len(np_open3d_colors) > 0:
            # Convert RGB colors (0-1 range) to packed rgb format
            rgb_packed = np.zeros(np_open3d_colors.shape[0], dtype=np.float32)
            for i in range(np_open3d_colors.shape[0]):
                r = int(np_open3d_colors[i, 0] * 255.0)
                g = int(np_open3d_colors[i, 1] * 255.0)
                b = int(np_open3d_colors[i, 2] * 255.0)
                rgb_packed[i] = struct.unpack('f', struct.pack('BBBB', b, g, r, 0))[0]
            
            # Create structured array with correct offsets
            cloud_data = np.zeros(np_open3d_cloud.shape[0], 
                                 dtype=[('x', np.float32), 
                                       ('y', np.float32), 
                                       ('z', np.float32),
                                       ('padding', np.uint64),  # 8-byte padding
                                       ('rgb', np.float32)])
            
            cloud_data['x'] = np_open3d_cloud[:, 0]
            cloud_data['y'] = np_open3d_cloud[:, 1]
            cloud_data['z'] = np_open3d_cloud[:, 2]
            cloud_data['rgb'] = rgb_packed
        else:
            # Create structured array with correct offsets but no RGB
            cloud_data = np.zeros(np_open3d_cloud.shape[0], 
                                 dtype=[('x', np.float32), 
                                       ('y', np.float32), 
                                       ('z', np.float32),
                                       ('padding', np.uint64)])  # 8-byte padding
            
            cloud_data['x'] = np_open3d_cloud[:, 0]
            cloud_data['y'] = np_open3d_cloud[:, 1]
            cloud_data['z'] = np_open3d_cloud[:, 2]
        
        # Define fields with correct offsets
        fields = [
            PointField(name='x', offset=0, datatype=ros_dtype, count=1),
            PointField(name='y', offset=4, datatype=ros_dtype, count=1),
            PointField(name='z', offset=8, datatype=ros_dtype, count=1)
        ]
        
        if np_open3d_colors is not None and len(np_open3d_colors) > 0:
            fields.append(PointField(name='rgb', offset=16, datatype=ros_dtype, count=1))
        
        # Convert structured array to bytes
        data = cloud_data.tobytes()
        
        # Determine if this is an organized point cloud
        if np_open3d_cloud.shape[0] == 640 * 480:  # If it matches your dimensions
            height = 480
            width = 640
        else:
            height = 1
            width = np_open3d_cloud.shape[0]
        
        header = Header(frame_id=frame_id)
        return PointCloud2(
            header=header,
            height=height,
            width=width,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=point_step,
            row_step=point_step * width,
            data=data
        )

    def O3DPointCloud_to_ROSpc2(self, open3d_cloud, frame_id="map"):
        np_open3d_cloud_coordinates, np_open3d_cloud_colors = self.O3DPointCloud_to_nparray(open3d_cloud)
        
        return self.nparray_to_ROSpc2(
            np_open3d_cloud_coordinates, 
            frame_id, 
            np_open3d_cloud_colors if len(np_open3d_cloud_colors) > 0 else None
        )

    def ROSpc2_to_nparray(self, pc2):
        # This method handles the direct byte access to account for specific offsets
        num_points = pc2.width * pc2.height
        
        # Create a numpy array from the byte data
        # Using np.frombuffer is faster than iterating through bytes
        cloud_array = np.frombuffer(pc2.data, dtype=np.uint8).reshape(num_points, pc2.point_step)
        
        # Extract x, y, z fields (assuming float32 type)
        x_offset = y_offset = z_offset = rgb_offset = -1
        
        # Find the correct offsets from the fields
        for field in pc2.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name == 'rgb':
                rgb_offset = field.offset
        
        if -1 in [x_offset, y_offset, z_offset]:
            print("Error: Could not find x, y, z fields in point cloud")
            return None, None, -1, -1
        
        # Extract xyz coordinates
        xyz = np.zeros((num_points, 3), dtype=np.float32)
        
        xyz[:, 0] = np.frombuffer(cloud_array[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
        xyz[:, 1] = np.frombuffer(cloud_array[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
        xyz[:, 2] = np.frombuffer(cloud_array[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
        
        # Remove points with NaN values
        valid_mask = ~np.isnan(xyz).any(axis=1)
        xyz = xyz[valid_mask]
        
        colors = None
        if rgb_offset != -1:
            # Extract RGB values and convert to Open3D format
            rgb_raw = np.frombuffer(cloud_array[valid_mask, rgb_offset:rgb_offset+4].tobytes(), dtype=np.float32)
            colors = np.zeros((len(rgb_raw), 3), dtype=np.float32)
            
            # Unpack RGB values
            for i in range(len(rgb_raw)):
                rgb = struct.unpack('BBBB', struct.pack('f', rgb_raw[i]))
                colors[i, 0] = rgb[2] / 255.0  # R
                colors[i, 1] = rgb[1] / 255.0  # G
                colors[i, 2] = rgb[0] / 255.0  # B
        
        return xyz, colors, -1, -1  # Not using intensity

    def nparray_to_O3DPointCloud(self, xyz, colors=None):
        open3d_cloud = o3d.geometry.PointCloud()

        # Find valid points (not NaN or inf)
        valid_idx = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[valid_idx]
        colors = colors[valid_idx]

        # Set points
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

        # Set colors if available
        if colors is not None and len(colors) > 0:
            open3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        return open3d_cloud
        
    def ROSpc2_to_O3DPointCloud(self, pc2):
        xyz, colors, _, _ = self.ROSpc2_to_nparray(pc2)
        if xyz is None:
            print("Failed to convert point cloud")
            return None, -1, -1
            
        return self.nparray_to_O3DPointCloud(xyz, colors), -1, -1