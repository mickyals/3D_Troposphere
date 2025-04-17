import argparse
import os
import torch
import pytorch_lightning as pl
import GPUtil as GPU

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from datetime import datetime
from data import AtmosphereDataset, AtmosphereIterableDataset
from torch.utils.data import DataLoader
from helpers import debug_print, set_device
from point_cloud_generator.pointcloudgenerator import PointCloudGenerator  # Ensure this is implemented
from models import *
import wandb

RENDER_TYPES = ['lon', 'lat', 'all']

def gpu_info():
    GPUs = GPU.getGPUs()
    gpu = GPUs[0] 
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    gpu_info()


# ---------------- Step 1: Argument Parsing ----------------
def get_parser():
    """Set up argument parsing for main.py"""
    parser = argparse.ArgumentParser(description="Train INR Model for Atmospheric Data")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--render_type", type=str, help="lat, lon, or all")
    parser.add_argument("--generate_pc", action="store_true", help="Generate point cloud after training")
    return parser.parse_args()

# ---------------- Step 2: Load Config ----------------
def load_config(config_path):
    """Loads the YAML config file using OmegaConf"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")
    return OmegaConf.load(config_path)

# ---------------- Step 3: Lightning DataModule ----------------
class AtmosphereDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config.dataset
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.prefetch_factor = self.config.prefetch_factor

    def setup(self, stage=None):
        """Initialize dataset (only training set)."""
        #debug_print()
        self.train_dataset =  AtmosphereIterableDataset(self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Crucial for IterableDataset
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def on_train_epoch_end(self):
        """Essential for multi-epoch training"""
        self.train_dataset.reset()

# ---------------- Step 3: Main Execution ----------------
if __name__ == "__main__":
    # Parse arguments
    args = get_parser()

    # Load YAML config
    config = load_config(args.config)

    # # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    print("Config Loaded Successfully")

    # # Instantiate DataModule with optimized settings
    data_module = AtmosphereDataModule(config)

    # # Instantiate Model
    target_str = config.model._target_
    model = INRModel(config).to(set_device())
    torch.set_float32_matmul_precision('medium')  # Set precision for matmul operations

    # # Callback setup
    model_name = config.get("run_name", "UnknownModel")
    run_name_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not args.generate_pc:
        callback_cfg = config.trainer.callbacks.ModelCheckpoint
        os.makedirs(callback_cfg.dirpath, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor=callback_cfg.monitor,
            mode=callback_cfg.mode,
            dirpath=callback_cfg.dirpath,
            save_top_k=callback_cfg.save_top_k,
            save_last=callback_cfg.save_last,
            filename= f"model_{model_name}_{run_name_date}"
        )

        # WandB configuration
        wandb_cfg = config.wandb

        wandb_logger = WandbLogger(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=f"{model_name}_{run_name_date}",
            log_model=True,
            config=OmegaConf.to_container(config, resolve=True)
        )

        # Optimized Trainer configuration
        trainer = pl.Trainer(
            max_epochs=config.trainer.max_epochs,
            accelerator="auto",  # Automatically uses GPU if available
            devices="auto",
            precision="16-mixed",  # Mixed precision training
            gradient_clip_val=config.trainer.get("gradient_clip", 0.5),  # Prevent exploding gradients
            deterministic=False,  # Deterministic training maintains reproducibility
            logger=wandb_logger,
            log_every_n_steps=10,
            enable_progress_bar=True,  # Disable if using in notebook
            enable_checkpointing=True,
            use_distributed_sampler=False,  # For IterableDataset
        )

        # Training with optimized data loader
        try:
            trainer.fit(model, datamodule=data_module)
        finally:
            torch.save(model.state_dict(), os.path.join(callback_cfg.dirpath, f'{model_name}_{run_name_date}_model.pt'))
            wandb.finish()  # Ensure clean exit

    # ---------------- Step 5: Generate Point Cloud (Optional) ----------------
    if args.generate_pc:
        debug_print()
        print("Generating point cloud from trained model...")

        render_type = args.render_type.lower()
        if render_type not in RENDER_TYPES:
            raise Exception(f"Invalid render type: must be {RENDER_TYPES}, got {render_type}")

        # Load trained model from checkpoint
        checkpoint_path = config.model_checkpoint  # Ensure this is defined in config
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        # Initialize PointCloudGenerator
        config.model._target_ = target_str
        pc_generator = PointCloudGenerator(model, config, device="cpu")
        pointcloud_filename = pc_generator.generate(render_type=render_type)

        print(f"Point cloud saved to {pointcloud_filename}")
