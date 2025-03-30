# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3Ds
import argparse
import pyvista as pv

def render_point_cloud(ply_file):
    # Load PLY file
    pcd = pv.read(ply_file)
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    print(pcd)
    # Check if temperature data exists
    if "temperature" in pcd.cell_data:
        temperature = np.asarray(pcd.cell_data["temperature"])
    else:
        print("Temperature data not found in the PLY file.")
        return
    
    # Plot using Matplotlib
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temperature, cmap='viridis', s=1)
    fig.colorbar(scatter, label="Temperature")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Geopotential Height")
    ax.set_title("3D Point Cloud with Temperature Colormap")
    print("done")
    plt.show()

if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser(description="Render a PLY point cloud with temperature colormap.")
    parser.add_argument("ply_file", type=str, help="Path to the PLY file.")
    args = parser.parse_args()
    render_point_cloud(args.ply_file)
