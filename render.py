
from data import *
from models import *
from point_cloud_generator import *
from pointcloudhelpers import *
from utils import *
from helpers import *
from main import *
import pyvista as pv 
import numpy as np

# Load YAML config
config = load_config("configs/config.yaml")

# # Set random seed for reproducibility
seed_everything(config.seed, workers=True)
print("Config Loaded Successfully")

# # Instantiate DataModule with optimized settings
data_module = AtmosphereDataModule(config)

# # Instantiate Model
target_str = config.model._target_
model = INRModel(config).to(set_device())

print("Generating point cloud from trained model...")

# Load trained model from checkpoint
checkpoint_path = config.model_checkpoint  # Ensure this is defined in config
print(checkpoint_path)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Initialize PointCloudGenerator
config.model._target_ = target_str
pc_generator = PointCloudGenerator(model, config, device="cpu")

for render_type in RENDER_TYPES:
    print(render_type)
    render_type = render_type.lower()
    if render_type not in RENDER_TYPES:
        raise Exception(f"Invalid render type: must be {RENDER_TYPES}, got {render_type}")
    pointcloud_filename = pc_generator.generate(model=target_str.split('.')[-1], render_type=render_type)

    print(f"Point cloud saved to {pointcloud_filename}")