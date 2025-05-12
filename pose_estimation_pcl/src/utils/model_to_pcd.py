#!/usr/bin/env python3
# filepath: obj_to_pcd_converter.py

import argparse
import numpy as np
import open3d as o3d
import os

def obj_to_pcd(obj_path, pcd_path, sample_points=10000, normals=True):
    """
    Convert .obj mesh file to .pcd point cloud file
    
    Args:
        obj_path: Path to the input .obj file
        pcd_path: Path to save the output .pcd file
        sample_points: Number of points to sample from the mesh
        normals: Whether to compute normals for the point cloud
    """
    print(f"Converting {obj_path} to {pcd_path}")
    
    # Load the mesh
    try:
        mesh = o3d.io.read_triangle_mesh(obj_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return False
    
    # Basic mesh info
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    
    # Sample points from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    
    # Compute normals if requested
    if normals:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Save as PCD
    try:
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Successfully saved point cloud with {len(pcd.points)} points to {pcd_path}")
        return True
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert OBJ files to PCD format")
    parser.add_argument("input", help="Path to input OBJ file")
    parser.add_argument("--output", help="Path to output PCD file (default: input_file_name.pcd)")
    parser.add_argument("--points", type=int, default=10000, help="Number of points to sample (default: 10000)")
    parser.add_argument("--no-normals", action="store_true", help="Skip normal estimation")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}.pcd"
    
    # Perform the conversion
    obj_to_pcd(args.input, args.output, args.points, not args.no_normals)
    
if __name__ == "__main__":
    main()