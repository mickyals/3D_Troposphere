import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr
import numpy as np
from helpers import debug_print




class AtmosphereDataset(Dataset):
    def __init__(self, config):
        """
        Initialize dataset from a configuration file.
        Args:
            config: OmegaConf config with dataset settings.
        """
        debug_print()
        self.nc_file = config.nc_file
        self.points_per_batch = config.get("points_per_batch", 500000)

        # open dataset
        self.ds = self._open_dataset()

        # Extract dimensions
        self.time_dim = len(self.ds["valid_time"])
        self.pressure_dim = len(self.ds["pressure_level"])
        self.lat_dim = len(self.ds["latitude"])
        self.lon_dim = len(self.ds["longitude"])

        # Convert coordinate variables to numpy arrays
        self.valid_time = self.ds["valid_time"].values
        self.pressure_levels = self.ds["pressure_level"].values
        self.latitudes = self.ds["latitude"].values
        self.longitudes = self.ds["longitude"].values

        # Total points per timestep
        self.total_points = self.pressure_dim * self.lat_dim * self.lon_dim
        self.batches_per_timestep = self.total_points // self.points_per_batch


    def _open_dataset(self):
        """Open the NetCDF file with chunking along time dimension"""
        debug_print()
        return xr.open_dataset(
            self.nc_file,
            chunks={"valid_time" :1},
            engine="netcdf4",
            decode_times=False
        )

    def __len__(self):
        debug_print()
        """Total batches in one epoch = time_dim * batches_per_timestep"""
        return self.time_dim * self.batches_per_timestep



    def __getitem__(self, index):
        """
        Retrieve all (lat, lon, pressure) samples for a given time step,
        then flatten and randomly shuffle them. In this version, the network
        inputs are:
             [normalized geopotential height, x_coord, y_coord, z_coord],
        while PDE inputs include the real pressure, geopotential height,
        specific humidity, and the base geopotential height (at 1000 mb).
        """
        # Select the current time step using the valid_time coordinate.
        debug_print()
        # Determine which time step this batch belongs to
        time_idx = index // self.batches_per_timestep
        batch_offset = index % self.batches_per_timestep

        # Select the current time step
        current_timestep = self.valid_time[time_idx]

        # Extract the 3D fields for this time step
        temp = self.ds["t"].sel(valid_time=current_timestep).values
        geopotential_height = self.ds["z"].sel(valid_time=current_timestep).values
        specific_humidity = self.ds["q"].sel(valid_time=current_timestep).values
        gh_base_2d = self.ds["z"].sel(valid_time=current_timestep, pressure_level=1000).values

        # Create spatial grids
        pressure_grid, lat_grid, lon_grid = np.meshgrid(
            self.pressure_levels, self.latitudes, self.longitudes, indexing="ij"
        )

        # Compute Cartesian coordinates
        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_grid - 180)
        x_coord = np.cos(lat_rad) * np.cos(lon_rad)
        y_coord = np.cos(lat_rad) * np.sin(lon_rad)
        z_coord = np.sin(lat_rad)

        # Normalization
        K_min, K_max = 183, 330
        gh_min, gh_max = -428.6875, 48664.082

        temp_norm = 2 * (temp - K_min) / (K_max - K_min) - 1
        gh_norm = 2 * (geopotential_height - gh_min) / (gh_max - gh_min) - 1

        # Flatten arrays
        inputs_array = np.stack([
            gh_norm.flatten(),
            x_coord.flatten(),
            y_coord.flatten(),
            z_coord.flatten()
        ], axis=1)

        target_array = temp_norm.flatten()

        # PDE inputs
        pde_pressure = pressure_grid.flatten()
        pde_geopotential = geopotential_height.flatten()
        pde_specific_humidity = specific_humidity.flatten()
        gh_base_flat = gh_base_2d.flatten()
        gh_base_repeated = np.tile(gh_base_flat, self.pressure_dim)

        # **Sequential Sampling: Select points in order**
        batch_start = batch_offset * self.points_per_batch
        batch_end = batch_start + self.points_per_batch
        batch_indices = np.arange(batch_start, batch_end)

        # Sample the batch (NO SHUFFLING)
        inputs_tensor = torch.tensor(inputs_array[batch_indices], dtype=torch.float32)
        target_tensor = torch.tensor(target_array[batch_indices], dtype=torch.float32).unsqueeze(1)

        pde_inputs_dict = {
            "pressure_level": torch.tensor(pde_pressure[batch_indices], dtype=torch.float32).unsqueeze(1),
            "geopotential_height": torch.tensor(pde_geopotential[batch_indices], dtype=torch.float32).unsqueeze(1),
            "specific_humidity": torch.tensor(pde_specific_humidity[batch_indices], dtype=torch.float32).unsqueeze(1),
            "base_geopotential_height": torch.tensor(gh_base_repeated[batch_indices], dtype=torch.float32).unsqueeze(1)
        }

        return {
            "inputs": inputs_tensor,
            "target": target_tensor,
            "pde_inputs": pde_inputs_dict
        }


class AtmosphereIterableDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.nc_file = config.nc_file
        self.points_per_batch = config.points_per_batch
        self.K_min, self.K_max = 183, 330
        self.gh_min, self.gh_max = -428.6875, 48664.082

    def __iter__(self):
        with xr.open_dataset(self.nc_file, engine="netcdf4") as ds:
            # Iterate over timesteps
            for time_idx in range(len(ds.valid_time)):
                # Process only this timestep (lazy loading)
                temp = ds.t.isel(valid_time=time_idx).values
                gh = ds.z.isel(valid_time=time_idx).values
                q = ds.q.isel(valid_time=time_idx).values
                gh_base = ds.z.sel(pressure_level=1000).isel(valid_time=time_idx).values

                # Meshgrid for coordinates (only once)
                pressure_levels = ds.pressure_level.values
                latitudes = ds.latitude.values
                longitudes = ds.longitude.values
                pressure_grid, lat_grid, lon_grid = np.meshgrid(pressure_levels, latitudes, longitudes, indexing="ij")

                # Convert to Cartesian coordinates (only once)
                lat_rad = np.deg2rad(lat_grid)
                lon_rad = np.deg2rad(lon_grid - 180)
                x_flat = (np.cos(lat_rad) * np.cos(lon_rad)).astype(np.float32).ravel()
                y_flat = (np.cos(lat_rad) * np.sin(lon_rad)).astype(np.float32).ravel()
                z_flat = np.sin(lat_rad).astype(np.float32).ravel()
                pressure_flat = pressure_grid.astype(np.float32).ravel()

                # Normalize temperature and geopotential height
                temp_norm = (2 * (temp - self.K_min) / (self.K_max - self.K_min) - 1).astype(np.float32).ravel()
                gh_norm = (2 * (gh - self.gh_min) / (self.gh_max - self.gh_min) - 1).astype(np.float32).ravel()
                q_flat = q.astype(np.float32).ravel()
                gh_base_flat = np.repeat(gh_base, len(pressure_levels)).astype(np.float32)

                total_points = temp_norm.shape[0]

                # **Batch processing instead of full loading**
                for i in range(0, total_points, self.points_per_batch):
                    batch_indices = slice(i, min(i + self.points_per_batch, total_points))  # Safe indexing
                    yield {
                        "inputs": torch.tensor(np.column_stack([
                            gh_norm[batch_indices], x_flat[batch_indices],
                            y_flat[batch_indices], z_flat[batch_indices]
                        ]), dtype=torch.float32),

                        "target": torch.tensor(temp_norm[batch_indices], dtype=torch.float32).unsqueeze(1),

                        "pde_inputs": {
                            "pressure_level": torch.tensor(pressure_flat[batch_indices], dtype=torch.float32).unsqueeze(
                                1),
                            "specific_humidity": torch.tensor(q_flat[batch_indices], dtype=torch.float32).unsqueeze(1),
                            "base_geopotential_height": torch.tensor(gh_base_flat[batch_indices],
                                                                     dtype=torch.float32).unsqueeze(1),
                        }
                    }