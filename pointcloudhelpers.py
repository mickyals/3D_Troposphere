import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature



def save_point_cloud_ply_latlon(point_cloud, model='MLP', filename="point_cloud.ply", render_type='all'):

    # Convert dictionary arrays to numpy arrays.
    lat = np.array(point_cloud["lat"])
    lon = np.array(point_cloud["lon"])
    gh = np.array(point_cloud["gh"])
    temperature = np.array(point_cloud["temperature"])

    # Instead of converting to Cartesian, we use the lat-lon grid directly.
    # x coordinate = longitude, y coordinate = latitude, z coordinate = geopotential height.
    points = np.column_stack([lon, lat, gh])

    render_point_cloud(points, temperature, model=model, render_type=render_type)

    # Create a PyVista PolyData object from the points.
    cloud = pv.PolyData(points)

    # Attach temperature as a scalar attribute.
    cloud["temperature"] = temperature

    print(cloud)

    # Save the PolyData to a binary PLY file.
    cloud.save(filename)
    print(f"Point cloud saved to {filename}")

def render_point_cloud(points, temperature, model='MLP', render_type='all'):
    axis_x, axis_y, axis_z = points[:, 0], points[:, 1], points[:, 2]

    d = {
        'lon': [points[:, 1], points[:, 2], points[:, 0]],
        'lat': [points[:, 0], points[:, 2], points[:, 1]],
        'all': [points[:, 0], points[:, 1], points[:, 2]]
    }

    text = render_type
    if render_type == 'all':
        text = 'GPH'
    axis_x, axis_y, axis_z = d[render_type]

    num_plot = 6

    axis_z_sampled = np.sort(np.unique(axis_z))
    axis_z_sampled = axis_z_sampled[::int(np.floor(len(axis_z_sampled)/num_plot))]

    plot_per_row = 3
    if render_type == 'all':
        fig, axes = plt.subplots(int(num_plot/plot_per_row), plot_per_row, figsize=(18, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(int(num_plot/plot_per_row), plot_per_row, figsize=(18, 8))
    axes = axes.flatten()
    for i, level in enumerate(axis_z_sampled[:num_plot]):
        ax = axes[i]
        index = np.where(axis_z == level)[0]
        if render_type == 'all':
            im = ax.scatter(axis_x[index], axis_y[index], c=temperature[index], s=0.5, transform=ccrs.PlateCarree(), cmap="inferno")
            ax.coastlines()
        else:
            im = ax.scatter(axis_x[index], axis_y[index], c=temperature[index], s=0.5, cmap="inferno")
        ax.set_title(f"{text} {np.round(level, 2)}", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
    # fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.02, left=0.03, right=0.97)
    plt.colorbar(im, ax=axes, orientation="horizontal", label=f"t Kelvin", aspect=80)
    fig.suptitle(f'World tempurature prediction {text} {np.round(np.min(axis_z_sampled), 2)} to {text} {np.round(np.max(axis_z_sampled), 2)}', size=30)
    plt.savefig(f'images/World_{model}_{text}.png')
    plt.show()
    print(f'Plot Saved: images/World_{model}_{text}.png')
