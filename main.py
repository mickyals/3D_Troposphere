import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from datetime import datetime
from data import  AtmosphereIterableDataset, AtmosphereDataset
from torch.utils.data import DataLoader
from helpers import debug_print, set_device
from point_cloud_generator.pointcloudgenerator import PointCloudGenerator  # Ensure this is implemented
from models import *
import wandb

# ---------------- Step 1: Argument Parsing ----------------
def get_parser():
    """Set up argument parsing for main.py"""
    parser = argparse.ArgumentParser(description="Train INR Model for Atmospheric Data")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
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
        self.shuffle = self.config.shuffle

    def setup(self, stage=None):
        """Initialize dataset (only training set)."""
        debug_print()
        self.train_dataset =  AtmosphereIterableDataset(self.config)

    def train_dataloader(self):
        """Returns DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            persistent_workers=False
        )

# ---------------- Step 3: Main Execution ----------------
if __name__ == "__main__":
    # Parse arguments
    args = get_parser()

    # Load YAML config
    config = load_config(args.config)

    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)
    print("Config Loaded Successfully")

    # Instantiate DataModule with optimized settings
    data_module = AtmosphereDataModule(config)

    # Instantiate Model
    model = INRModel(config).to(set_device())

    # Callback setup
    callback_cfg = config.trainer.callbacks.INRLoggerCallback
    logger_callback = INRLoggerCallback(
        monitor_metrics=callback_cfg.monitor_metrics,
        mode=callback_cfg.mode,
        save_path=callback_cfg.save_path
    )

    # WandB configuration
    wandb_cfg = config.wandb
    run_name_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = config.get("run_name", "UnknownModel")

    wandb_logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=f"INR_Training_{model_name}_{run_name_date}",
        log_model=True
    )

    # Optimized Trainer configuration
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto",
        precision="16-mixed",  # Mixed precision training
        gradient_clip_val=config.trainer.get("gradient_clip", 0.5),  # Prevent exploding gradients
        deterministic=False,  # Faster but maintains reproducibility
        callbacks=[logger_callback],
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
        wandb.finish()  # Ensure clean exit

    # ---------------- Step 5: Generate Point Cloud (Optional) ----------------
    if args.generate_pc:
        debug_print()
        print("Generating point cloud from trained model...")

        # Load trained model from checkpoint
        checkpoint_path = config.model_checkpoint  # Ensure this is defined in config
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        # Initialize PointCloudGenerator
        pc_generator = PointCloudGenerator(config, device="cpu")
        pointcloud_filename = pc_generator.generate()

        print(f"Point cloud saved to {pointcloud_filename}")
