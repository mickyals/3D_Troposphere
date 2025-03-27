import importlib
import torch
import numpy as np
from helpers import set_device, debug_print
from pointcloudhelpers import save_point_cloud_ply_latlon



class PointCloudGenerator:
    def __init__(self, model, config, device=None):
        debug_print()
        self.config = config
        self.device = set_device()

        target_str = self.config.model.pop("_target_")
        self.target_str = target_str

        # load model
        self.model = model.to(self.device)

        # sampling for point cloud
        # self.num_points = config.pointcloud.num_points
        self.lat_range = config.pointcloud.lat_range  # (min, max)
        self.lon_range = config.pointcloud.lon_range  # (min, max)
        self.gh_range = config.pointcloud.geopotential_height_range  # (min, max)

        # Normalization bounds
        self.K_min, self.K_max = 183, 330  # Temperature bounds in Kelvin
        self.gh_min, self.gh_max = -428.6875, 48664.082  # based on the original range of the data

    # def load_model(self):
    #     """ load model from config"""
    #     debug_print()
    #     model_cfg = self.config.model
    #     checkpoint_path = self.config.model_checkpoint  # Path to the saved weights

    #     target_str = model_cfg.pop("_target_")
    #     self.target_str = target_str
    #     module_path, class_name = target_str.rsplit(".", 1)
    #     module = importlib.import_module(module_path)
    #     model_class = getattr(module, class_name)

    #     # instantiate model
    #     model = model_class(model_cfg).to(self.device)

    #     # load weights
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
    #     model.eval()  # Set to evaluation mode
    #     print(f"Loaded model from {checkpoint_path}")

    #     return model

    def sample_points(self):
        debug_print()
        # Define number of points explicitly from config
        num_lats = self.config.pointcloud.resolution.num_lats
        num_lons = self.config.pointcloud.resolution.num_lons
        num_heights = self.config.pointcloud.resolution.num_geopotential_heights  # Vertical resolution

        # Sample latitude, longitude, and geopotential height at specified resolutions
        lat_samples = np.linspace(self.lat_range[0], self.lat_range[1], num_lats)
        lon_samples = np.linspace(self.lon_range[0], self.lon_range[1], num_lons)
        gh_samples = np.linspace(self.gh_range[0], self.gh_range[1], num_heights)

        # Convert lat and lon degrees to radians
        lat_rad = np.radians(lat_samples)
        lon_rad = np.radians(lon_samples)

        # Create a structured 3D grid of (lat, lon, gh)
        lat_grid, lon_grid, gh_grid = np.meshgrid(lat_samples, lon_samples, gh_samples, indexing="ij")

        # Convert lat, lon to Cartesian coordinates
        x = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
        y = np.cos(np.radians(lat_grid)) * np.sin(np.radians(lon_grid))
        z = np.sin(np.radians(lat_grid))

        # Normalize the geopotential height to [-1, 1]
        gh_norm = 2 * (gh_grid - self.gh_min) / (self.gh_max - self.gh_min) - 1

        # Stack features: INR model expects [x, y, z, gh_norm]
        inputs = np.stack([x.flatten(), y.flatten(), z.flatten(), gh_norm.flatten()], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        return inputs_tensor, lat_grid.flatten(), lon_grid.flatten(), gh_grid.flatten()

    def generate(self, render_type='all'):
        debug_print()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
        # Sample points based on config.
        inputs_tensor, lat_samples, lon_samples, gh_samples = self.sample_points()

        # Use the model to predict temperature. The model returns normalized temperature.
        with torch.no_grad():
            temp_pred_norm = self.model(inputs_tensor)
            # Unnormalize temperature from [-1, 1] to [K_min, K_max]
            temp_real = (temp_pred_norm + 1) / 2 * (self.K_max - self.K_min) + self.K_min

        point_cloud = {
            "lat": lat_samples,
            "lon": lon_samples,
            "gh": gh_samples,
            "temperature": temp_real.cpu().numpy()
        }

        # Extract model class name from the _target_ string.
        model_target = self.target_str
        _, class_name = model_target.rsplit(".", 1)
        filename = f"{class_name}_point_cloud.ply"
        save_point_cloud_ply_latlon(point_cloud, filename=filename, render_type=render_type)

        return filename