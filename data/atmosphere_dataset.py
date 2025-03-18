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
        return self.time_dim # 24 batches of data - 24 images of the atmosphere - 24 time stamps



    def __getitem__(self, time_idx):
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
        current_timestep = self.valid_time[time_idx]

        # Extract the 3D fields for this time step (shape: [Pressure_level, Lat, Lon])
        temp = self.ds["t"].sel(valid_time=current_timestep).values
        geopotential_height = self.ds["z"].sel(valid_time=current_timestep).values
        specific_humidity = self.ds["q"].sel(valid_time=current_timestep).values

        # Extract the base geopotential height at 1000 mb directly.
        # This returns a 2D array of shape (L, Lo).
        gh_base_2d = self.ds["z"].sel(valid_time=current_timestep, pressure_level=1000).values

        # Create spatial grids for pressure level, latitude, longitude.
        pressure_grid, lat_grid, lon_grid = np.meshgrid(
            self.pressure_levels, self.latitudes, self.longitudes, indexing="ij"
        )

        # Compute Cartesian coordinates on the fly from lat/lon.
        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_grid - 180)  # Convert [0,360] to [-180,180]
        x_coord = np.cos(lat_rad) * np.cos(lon_rad)
        y_coord = np.cos(lat_rad) * np.sin(lon_rad)
        z_coord = np.sin(lat_rad)

        # Normalization constants.
        K_min, K_max = 183, 330                # Temperature in Kelvin.
        gh_min, gh_max = -428.6875, 48664.082    # Geopotential height range for dataset (m).

        # Normalize temperature and geopotential height.
        temp_norm = 2 * (temp - K_min) / (K_max - K_min) - 1
        gh_norm = 2 * (geopotential_height - gh_min) / (gh_max - gh_min) - 1

        # Build network inputs: use normalized geopotential height and Cartesian coordinates.
        inputs_array = np.stack([
            gh_norm.flatten(),  # Normalized geopotential height (input)
            x_coord.flatten(),  # x coordinate
            y_coord.flatten(),  # y coordinate
            z_coord.flatten()   # z coordinate
        ], axis=1)  # Shape: (N, 4), where N = pressure_dim * lat_dim * lon_dim

        # The target is the normalized temperature.
        target_array = temp_norm.flatten()  # Shape: (N,)

        # Build PDE inputs using real (unnormalized) values.
        pde_pressure = pressure_grid.flatten()             # Real pressure in hPa.
        pde_geopotential = geopotential_height.flatten()     # Real geopotential height in m.
        pde_specific_humidity = specific_humidity.flatten()  # Specific humidity.

        # Compute base geopotential height for each spatial column.
        # gh_base_2d is shape (L, Lo); flatten it to get (L*Lo,)
        gh_base_flat = gh_base_2d.flatten()
        # Duplicate (tile) this array for each pressure level.
        P = self.pressure_dim
        gh_base_repeated = np.tile(gh_base_flat, P)  # Shape: (P * L * Lo,)

        # Build the PDE inputs dictionary (we keep each variable separate).
        pde_inputs_dict = {
            "pressure_level": torch.tensor(pde_pressure, dtype=torch.float32).unsqueeze(1),
            "geopotential_height": torch.tensor(pde_geopotential, dtype=torch.float32).unsqueeze(1),
            "specific_humidity": torch.tensor(pde_specific_humidity, dtype=torch.float32).unsqueeze(1),
            "base_geopotential_height": torch.tensor(gh_base_repeated, dtype=torch.float32).unsqueeze(1)
        }

        # Shuffle the spatial points uniformly.
        N = inputs_array.shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)
        inputs_array = inputs_array[indices]
        target_array = target_array[indices]
        # Also shuffle the PDE arrays.
        pde_pressure = pde_pressure[indices]
        pde_geopotential = pde_geopotential[indices]
        pde_specific_humidity = pde_specific_humidity[indices]
        gh_base_repeated = gh_base_repeated[indices]

        # Convert to torch tensors.
        inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32).unsqueeze(1)
        # Rebuild PDE inputs dictionary with shuffled values.
        pde_inputs_dict = {
            "pressure_level": torch.tensor(pde_pressure, dtype=torch.float32).unsqueeze(1),
            "geopotential_height": torch.tensor(pde_geopotential, dtype=torch.float32).unsqueeze(1),
            "specific_humidity": torch.tensor(pde_specific_humidity, dtype=torch.float32).unsqueeze(1),
            "base_geopotential_height": torch.tensor(gh_base_repeated, dtype=torch.float32).unsqueeze(1)
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
        # Open dataset INSIDE iterator
        with xr.open_dataset(self.nc_file, engine="netcdf4") as ds:
            # Precompute static grids
            pressure_grid, lat_grid, lon_grid = np.meshgrid(
                ds.pressure_level.values,
                ds.latitude.values,
                ds.longitude.values,
                indexing="ij"
            )

            print(ds, type(ds))

            # Convert to Cartesian
            lat_rad = np.deg2rad(lat_grid)
            lon_rad = np.deg2rad(lon_grid - 180)
            x_flat = (np.cos(lat_rad) * np.cos(lon_rad)).ravel()
            y_flat = (np.cos(lat_rad) * np.sin(lon_rad)).ravel()
            z_flat = np.sin(lat_rad).ravel()
            pressure_flat = pressure_grid.ravel()

            # Load time-dependent data
            temp_data = ds.t.load().values
            z_data = ds.z.load().values
            q_data = ds.q.load().values
            # q_data = np.full(z_data.shape, 5e-9)
            gh_base_data = ds.z.sel(pressure_level=1000).load().values

            for time_idx in range(len(ds.valid_time)):
                # Process timestep
                temp = temp_data[time_idx].ravel()
                gh = z_data[time_idx].ravel()
                q = q_data[time_idx].ravel()
                gh_base = gh_base_data[time_idx].ravel()

                # Normalize
                temp_norm = 2 * (temp - self.K_min) / (self.K_max - self.K_min) - 1
                gh_norm = 2 * (gh - self.gh_min) / (self.gh_max - self.gh_min) - 1

                # Build batch
                inputs = np.column_stack([gh_norm, x_flat, y_flat, z_flat])
                indices = np.random.choice(inputs.shape[0], self.points_per_batch, False)

                yield {
                    "inputs": torch.as_tensor(inputs[indices], dtype=torch.float32),
                    "target": torch.as_tensor(temp_norm[indices], dtype=torch.float32).unsqueeze(1),
                    "pde_inputs": {
                        "pressure_level": torch.as_tensor(pressure_flat[indices], dtype=torch.float32).unsqueeze(1),
                        "specific_humidity": torch.as_tensor(q[indices], dtype=torch.float32).unsqueeze(1),
                        "base_geopotential_height": torch.as_tensor(
                            np.repeat(gh_base, len(ds.pressure_level))[indices],
                            dtype=torch.float32
                        ).unsqueeze(1),
                    }
                }