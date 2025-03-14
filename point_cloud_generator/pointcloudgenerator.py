import importlib
import torch
import numpy as np
from helpers import set_device, debug_print
from pointcloudhelpers import save_point_cloud_ply_latlon



class PointCloudGenerator:
    def __init__(self, config, device=None):
        debug_print()
        self.config = config
        self.device = set_device()

        # load model
        self.model = self.load_model()

        # sampling for point cloud
        self.num_points = config.pointcloud.num_points
        self.lat_range = config.pointcloud.lat_range  # (min, max)
        self.lon_range = config.pointcloud.lon_range  # (min, max)
        self.gh_range = config.pointcloud.geopotential_height_range  # (min, max)

        # Normalization bounds
        self.K_min, self.K_max = 183, 330  # Temperature bounds in Kelvin
        self.gh_min, self.gh_max = -428.6875, 48664.082  # based on the original range of the data

    def load_model(self):
        """ load model from config"""
        debug_print()
        model_cfg = self.config.model
        checkpoint_path = self.config.model_checkpoint  # Path to the saved weights

        module_path, class_name = model_cfg._target_.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # instantiate model
        model = model_class(model_cfg).to(self.device)

        # load weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()  # Set to evaluation mode
        print(f"Loaded model from {checkpoint_path}")

        return model

    def sample_points(self):
        debug_print()
        lat_samples = np.random.uniform(self.lat_range[0], self.lat_range[1], self.num_points)
        lon_samples = np.random.uniform(self.lon_range[0], self.lon_range[1], self.num_points)
        gh_samples = np.random.uniform(self.gh_range[0], self.gh_range[1], self.num_points)

        # convert to lat and lon degrees to radians
        lat_rad = np.radians(lat_samples)
        lon_rad = np.radians(lon_samples)

        # Convert lat, lon to Cartesian coordinates.
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Normalize the geopotential height to [-1, 1].
        gh_norm = 2 * (gh_samples - self.gh_min) / (self.gh_max - self.gh_min) - 1

        # Stack features: note that the INR model expects inputs [x, y, z, gh_norm].
        inputs = np.stack([x, y, z, gh_norm], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)

        return inputs_tensor, lat_samples, lon_samples, gh_samples

    def generate(self):
        debug_print()
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
        model_target = self.config.model._target_
        _, class_name = model_target.rsplit(".", 1)
        filename = f"{class_name}_point_cloud.ply"
        save_point_cloud_ply_latlon(point_cloud, filename=filename)

        return filename