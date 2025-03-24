import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature



def save_point_cloud_ply_latlon(point_cloud, filename="point_cloud.ply"):

    # Convert dictionary arrays to numpy arrays.
    lat = np.array(point_cloud["lat"])
    lon = np.array(point_cloud["lon"])
    gh = np.array(point_cloud["gh"])
    temperature = np.array(point_cloud["temperature"])

    # Instead of converting to Cartesian, we use the lat-lon grid directly.
    # x coordinate = longitude, y coordinate = latitude, z coordinate = geopotential height.
    points = np.column_stack([lon, lat, gh])

    render_point_cloud(points, temperature)

    # Create a PyVista PolyData object from the points.
    cloud = pv.PolyData(points)

    # Attach temperature as a scalar attribute.
    cloud["temperature"] = temperature

    print(cloud)

    # Save the PolyData to a binary PLY file.
    cloud.save(filename)
    print(f"Point cloud saved to {filename}")

def render_point_cloud(points, temperature):
    # # Plot using Matplotlib
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temperature, cmap='inferno', s=1)
    # fig.colorbar(scatter, label="Temperature")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Latitude")
    # ax.set_zlabel("Geopotential Height")
    # ax.set_title("3D Point Cloud with Temperature Colormap")
    # print("done")
    # plt.savefig('test.png')

    fig, axes = plt.subplots(5, 10, figsize=(18, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()
    gh = np.linspace(0, 20000, 50)
    for i, level in enumerate(gh):
        ax = axes[i]
        index = np.where(points[:, 2] == level)[0]
        im = ax.scatter(points[index, 0], points[index, 1], c=temperature[index], transform=ccrs.PlateCarree(), cmap="inferno")
        ax.coastlines()
        ax.set_title(f"{np.round(level, 2)} hPa", fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.colorbar(im, ax=axes, orientation="horizontal", label=f"t Kelvin")
    plt.title('Toronto 0 to 20000')
    plt.savefig('Toronto.png')
