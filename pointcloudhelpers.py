import pyvista as pv
import numpy as np



def save_point_cloud_ply_latlon(point_cloud, filename="point_cloud.ply"):

    # Convert dictionary arrays to numpy arrays.
    lat = np.array(point_cloud["lat"])
    lon = np.array(point_cloud["lon"])
    gh = np.array(point_cloud["gh"])
    temperature = np.array(point_cloud["temperature"]).flatten()

    # Instead of converting to Cartesian, we use the lat-lon grid directly.
    # x coordinate = longitude, y coordinate = latitude, z coordinate = geopotential height.
    points = np.column_stack([lon, lat, gh])

    # Create a PyVista PolyData object from the points.
    cloud = pv.PolyData(points)

    # Attach temperature as a scalar attribute.
    cloud["temperature"] = temperature

    # Save the PolyData to a binary PLY file.
    cloud.save(filename)
    print(f"Point cloud saved to {filename}")