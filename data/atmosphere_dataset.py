import torch
from torch.utils.data import IterableDataset
import xarray as xr
import numpy as np
from helpers import debug_print, set_device

class AtmosphereDataset(IterableDataset):
    def __init__(self, config):
        """
        Args:
            config: Configuration (e.g., OmegaConf dict) with keys:
              - nc_file: path to the preprocessed NetCDF file
              - batch_size: number of points per mini-batch
        """
        debug_print()
        self.nc_file = config.nc_file
        self.batch_size = config.batch_size
        self.device = set_device()

        # Normalization ranges (for possible denormalization if needed)
        self.temp_range = (183.0, 330.0)         # for t (in Kelvin)
        self.gh_range = (-428.6875, 48664.082)     # for z (in meters)

        # Open file briefly to extract dimensions
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False) as ds:
            self.num_timesteps = len(ds.valid_time)
            self.num_pressure = len(ds.pressure_level)
            self.num_lat = len(ds.latitude)
            self.num_lon = len(ds.longitude)

        # Total points per time step
        self.points_per_timestep = self.num_pressure * self.num_lat * self.num_lon
        self.batches_per_timestep = int(np.ceil(self.points_per_timestep / self.batch_size))

        # Precompute static spatial grid and pressure array (on CPU)
        self._precompute_static_data()

    def _precompute_static_data(self):
        """Precompute the spatial grid from latitude and longitude and the pressure values."""
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False) as ds:
            lats = ds.latitude.values.astype(np.float32)  # shape (num_lat,)
            lons = ds.longitude.values.astype(np.float32)  # shape (num_lon,)
            pressures = ds.pressure_level.values.astype(np.float32)  # shape (num_pressure,)

        # Create 2D grid for spatial coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")  # shape (num_lon, num_lat)
        # Convert longitudes from [0, 360] to [-180, 180]
        lon_centered = lon_grid - 180.0

        # Convert to radians
        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_centered)

        # Compute Cartesian coordinates (spherical to Cartesian on unit sphere)
        cos_lat = np.cos(lat_rad)
        x = (cos_lat * np.cos(lon_rad)).astype(np.float32)
        y = (cos_lat * np.sin(lon_rad)).astype(np.float32)
        z = np.sin(lat_rad).astype(np.float32)

        # Flatten the 2D spatial grid (order: C-order, so that the fastest axis is longitude)
        x_flat = x.ravel()  # shape: (num_lat * num_lon,)
        y_flat = y.ravel()
        z_flat = z.ravel()

        # Tile the spatial coordinates for each pressure level.
        # Each time step has total points = num_pressure * num_lat * num_lon.
        self.static_grid = {
            "x": torch.from_numpy(np.tile(x_flat, self.num_pressure)),
            "y": torch.from_numpy(np.tile(y_flat, self.num_pressure)),
            "z": torch.from_numpy(np.tile(z_flat, self.num_pressure))
        }

        # Pressure values: for each pressure level, repeat for all spatial locations.
        pressure_repeated = np.repeat(pressures, self.num_lat * self.num_lon)
        self.static_pressure = torch.from_numpy(pressure_repeated)

    def _load_time_step(self, time_idx):
        """Load time-dependent variables for a given time step."""
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False) as ds:
            ts = ds.isel(valid_time=time_idx).load()  # load the full time slice

            # Raw variables (shape: [num_pressure, num_lat, num_lon])
            t = ts.t.values.astype(np.float32)
            z = ts.z.values.astype(np.float32)
            q = ts.q.values.astype(np.float32)
            # Normalized variables
            t_norm = ts.t_norm.values.astype(np.float32)
            z_norm = ts.z_norm.values.astype(np.float32)
            # Base geopotential: select slice at pressure level 1000 (shape: [num_lat, num_lon])
            gh_base = ts.z.sel(pressure_level=1000).values.astype(np.float32)

        # Flatten arrays (using C-order so that order is [pressure, lat, lon])
        time_data = {
            "t": torch.as_tensor(t.ravel(), dtype=torch.float32),
            "z": torch.as_tensor(z.ravel(), dtype=torch.float32),
            "q": torch.as_tensor(q.ravel(), dtype=torch.float32),
            "t_norm": torch.as_tensor(t_norm.ravel(), dtype=torch.float32),
            "z_norm": torch.as_tensor(z_norm.ravel(), dtype=torch.float32),
            # For gh_base: repeat the flattened (lat,lon) array for each pressure level
            "gh_base": torch.as_tensor(np.repeat(gh_base.ravel(), self.num_pressure), dtype=torch.float32)
        }
        return time_data

    def _create_batch(self, time_data, batch_idx):
        """Construct a mini-batch from loaded time_data."""
        start = batch_idx * self.batch_size
        end = start + self.batch_size

        # Inputs: [x, y, z, z_norm]
        # Static grid data are precomputed
        x_batch = self.static_grid["x"][start:end]
        y_batch = self.static_grid["y"][start:end]
        z_batch = self.static_grid["z"][start:end]
        # z_norm is dynamic per time step (loaded from NC)
        z_norm_batch = time_data["z_norm"][start:end]
        inputs = torch.stack([x_batch, y_batch, z_batch, z_norm_batch], dim=1)

        # Target: normalized temperature t_norm
        target = time_data["t_norm"][start:end].unsqueeze(1)

        # PDE inputs: raw z, q, t, and static pressure, plus base geopotential
        pde_inputs = {
            "pressure_level": self.static_pressure[start:end].unsqueeze(1),
            "z": time_data["z"][start:end].unsqueeze(1),
            "q": time_data["q"][start:end].unsqueeze(1),
            "t": time_data["t"][start:end].unsqueeze(1),
            "gh_base": time_data["gh_base"][start:end].unsqueeze(1)
        }
        return {"inputs": inputs, "target": target, "pde_inputs": pde_inputs}

    def __iter__(self):
        """Iterate over each time step and yield mini-batches."""
        for time_idx in range(self.num_timesteps):
            debug_print()
            time_data = self._load_time_step(time_idx)
            for batch_idx in range(self.batches_per_timestep):
                yield self._create_batch(time_data, batch_idx)
            # Optional: clear GPU cache after each time step
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def __len__(self):
        """Total number of mini-batches across all time steps."""
        return self.num_timesteps * self.batches_per_timestep


class AtmosphereIterableDataset(IterableDataset):
    def __init__(self, config):
        debug_print()
        self.nc_file = config.nc_file
        self.batch_size = config.batch_size
        self.device = set_device()

        # Open dataset briefly to get dimensions
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False) as ds:
            self.num_timesteps = len(ds.valid_time)
            self.num_pressure = len(ds.pressure_level)
            self.num_lat = len(ds.latitude)
            self.num_lon = len(ds.longitude)

        # Total points per time step
        self.points_per_timestep = self.num_pressure * self.num_lat * self.num_lon
        self.batches_per_timestep = int(np.ceil(self.points_per_timestep / self.batch_size))

        # Precompute static spatial grid and pressure array
        self._precompute_static_data()

    def _precompute_static_data(self):
        """Precompute static spatial grid and pressure levels."""
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False) as ds:
            lats = ds.latitude.values.astype(np.float32)
            lons = ds.longitude.values.astype(np.float32)


        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
        lon_centered = lon_grid - 180.0

        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_centered)

        cos_lat = np.cos(lat_rad)
        x = (cos_lat * np.cos(lon_rad)).astype(np.float32)
        y = (cos_lat * np.sin(lon_rad)).astype(np.float32)
        z = np.sin(lat_rad).astype(np.float32)

        x_flat = x.ravel()
        y_flat = y.ravel()
        z_flat = z.ravel()

        # Store static variables
        self.static_grid = {
            "x": np.tile(x_flat, self.num_pressure),
            "y": np.tile(y_flat, self.num_pressure),
            "z": np.tile(z_flat, self.num_pressure),
        }

    def _load_time_step(self, time_idx):
        """Load and normalize all variables for a single timestep."""
        with xr.open_dataset(self.nc_file, engine="netcdf4", decode_times=False, drop_variables=["z"]) as ds:
            ts = ds.isel(valid_time=time_idx).load()

            #z = ts.z.values.astype(np.float32).ravel()
            q = ts.q.values.astype(np.float32).ravel()
            t = ts.t.values.astype(np.float32).ravel()
            t_norm = ts.t_norm.values.astype(np.float32).ravel()
            gh_norm = ts.z_norm.values.astype(np.float32).ravel()
            p1_p2_ratio = ts.p1_p2_ratio.values.astype(np.float32).ravel()
            mean_T_v = ts.mean_T_v.values.astype(np.float32).ravel()
            delta_z = ts.delta_z.values.astype(np.float32).ravel()


        # **Shuffle all points within the time step**
        indices = np.random.permutation(len(t))

        return {
            "q": torch.from_numpy(q)[indices],
            "t": torch.from_numpy(t)[indices],
            "t_norm": torch.from_numpy(t_norm)[indices],
            "gh_norm": torch.from_numpy(gh_norm)[indices],
            "p1_p2_ratio": torch.from_numpy(p1_p2_ratio)[indices],
            "mean_T_v": torch.from_numpy(mean_T_v)[indices],
            "delta_z": torch.from_numpy(delta_z)[indices],
            "x": torch.from_numpy(self.static_grid["x"])[indices],
            "y": torch.from_numpy(self.static_grid["y"])[indices],
            "z": torch.from_numpy(self.static_grid["z"])[indices],
        }

    def _create_batch(self, time_data, batch_idx):
        """Create a mini-batch from the shuffled time step."""
        start = batch_idx * self.batch_size
        end = start + self.batch_size

        inputs = torch.stack([
            time_data["x"][start:end],
            time_data["y"][start:end],
            time_data["z"][start:end],
            time_data["gh_norm"][start:end],
        ], dim=1)

        target = time_data["t_norm"][start:end].unsqueeze(1)

        pde_inputs = {
            "mean_T_v": time_data["mean_T_v"][start:end].unsqueeze(1),
            "q": time_data["q"][start:end].unsqueeze(1),
            "delta_z": time_data["delta_z"][start:end].unsqueeze(1),
            "p1_p2_ratio": time_data["p1_p2_ratio"][start:end].unsqueeze(1),
            "t": time_data["t"][start:end].unsqueeze(1)
        }

        return {"inputs": inputs, "target": target, "pde_inputs": pde_inputs}

    def __iter__(self):
        """Iterate over **shuffled** time steps and yield mini-batches."""
        time_indices = np.random.permutation(self.num_timesteps)  # Shuffle time step order
        for time_idx in time_indices:
            #debug_print()
            time_data = self._load_time_step(time_idx)  # **Shuffled within time step**
            for batch_idx in range(self.batches_per_timestep):
                yield self._create_batch(time_data, batch_idx)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    #
    # def __len__(self):
    #     return self.num_timesteps * self.batches_per_timestep