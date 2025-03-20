import pytorch_lightning as pl
import torch
from torch import nn
from utils import instantiate_from_config
import time
import wandb
import os
from helpers import set_seed, debug_print, set_device




class INRModel(pl.LightningModule):
    def __init__(self, config, device=None):
        """
    Base PINN model that instantiates the INR network dynamically from config,
    and includes PDE-based losses (hydrostatic loss and hypsometric regularizer).
    """
        super().__init__()
        #debug_print()
        self.config = config.model
        self.net = instantiate_from_config(self.config)

        # Loss hyper params\
        self.data_weight = self.config.loss_config.data_weight
        #self.physics_weight = self.config.loss_config.physics_weight
        self.regularizer_weight = self.config.loss_config.regularizer_weight

        # Dynamically configure the loss function from config.
        # If no loss is provided, default to MSELoss.
        loss_cfg = self.config.get("loss_config", {})
        loss_type = loss_cfg.get("type", "MSELoss")
        loss_params = loss_cfg.get("params", {})
        loss_class = getattr(nn, loss_type, nn.MSELoss)
        self.loss = loss_class(**loss_params)

        #self.physics_loss = HydrostaticLoss()  # Instantiate physics loss module
        self.gh_min = -428.6875
        self.gh_max = 48664.082

        # these can be commented out if not need for your task
        self.GRAVITY = 9.80665 # gravitational acceleration
        self.Rd = 287 # dry air constant


    def forward(self, x):
        #debug_print()
        return self.net(x.to(self.device))

    def training_step(self, batch, batch_idx):
        # Keep inputs as float32 but enable gradients
        inputs = batch['inputs'].to(self.device, non_blocking=True)  # [norm_gh, x, y, z]
        target = batch['target'].to(self.device, non_blocking=True)
        pde_inputs = {k: v.to(self.device) for k, v in batch["pde_inputs"].items()}

        # Forward pass (predicts normalized temperature)
        pred_norm = self.forward(inputs)  # Uses normalized inputs

        # Data loss (on normalized values)
        data_loss = self.loss(pred_norm, target)

        # Convert to REAL VALUES with gradient tracking
        K_min, K_max = 183, 330

        # Denormalize while preserving gradients
        T_real_pred = (pred_norm + 1) / 2 * (K_max - K_min) + K_min


        total_loss = self.data_weight*data_loss +  self.regularizer_weight*"item"

        # logging into wandb
        self.log("train/data_loss", data_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train/physics_loss", physics_loss, on_step=True, on_epoch=True)
        self.log("train/physics_regulariser", physics_regulariser, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        #debug_print()
        # load the optimser configs
        optim_cfg = self.config.optimizer_config
        optim_class = getattr(torch.optim, optim_cfg.type, torch.optim.Adam) # default is adam

        # initialise the optimiser from config
        optimizer = optim_class(self.parameters(), lr=optim_cfg.learning_rate, **optim_cfg.params)

        # load scheduler
        sched_cfg = self.config.scheduler
        if sched_cfg and sched_cfg.type:
            scheduler_class = getattr(torch.optim.lr_scheduler, sched_cfg.type, torch.optim.lr_scheduler.CosineAnnealingLR)
            scheduler = scheduler_class(optimizer, **sched_cfg.params)

            # return optimizer and schedule
            # Set interval dynamically
            interval = sched_cfg.params.get("interval", "epoch")

            return [optimizer], [{"scheduler": scheduler, "interval": interval}]
        return optimizer # if there is no scheduler

    def compute_physics_reg(self, real_pred, pde_inputs):
        """ A loss function for designing a physics regularization term, by default the loss is the hypsometric function


        For each spatial point, the expected thickness between 1000 mb and the current pressure level p
        is computed as:

             Δz_expected = (Rd/g) * T_real_pred * ln(1000 / p)

        The actual thickness is:

             Δz_actual = base_geopotential_height - geopotential_height
        """

        #debug_print()
        # Extract PDE variables
        p = pde_inputs["pressure_level"]    # shape (N, 1), in hPa
        q = pde_inputs["q"]                 # shape (N, 1), in kg/kg
        z = pde_inputs["z"]                 # shape (N, 1), in meters
        gh_base = pde_inputs["gh_base"]  # shape (N, 1), in meters

        T_v = real_pred * (1 + 0.61 * q)

        # Compute expected thickness using the hypsometric equation.
        expected_thickness = (self.Rd / self.GRAVITY) * T_v * torch.log(1000.0 / p)

        # Compute actual thickness: difference between the base geopotential height and the current geopotential height.
        actual_thickness = z - gh_base

        # Compute the hypsometric loss as the mean squared error between expected and actual thickness.
        hypsometric_loss = torch.mean((actual_thickness - expected_thickness) ** 2)

        return hypsometric_loss

class HydrostaticLoss(nn.Module):
    def __init__(self, gravity=9.80665, Rd=287.0):
        super().__init__()
        self.gravity = gravity
        self.Rd = Rd
        self.register_buffer("hpa_to_pa", torch.tensor(100.0))  # For unit conversion

    def forward(self, T_real, gh_real, pde_inputs):


        z = pde_inputs["z"]
        q = pde_inputs["q"]
        T_v = T_real * (1 + 0.61 * q)  # (N,1)

        # Hydrostatic balance equation components
        p_pa = pde_inputs["pressure_level"] * self.hpa_to_pa  # (N,1)
        expected_dP_dz = - (p_pa * self.gravity) / (self.Rd * T_v)

        dP_dz = torch.autograd.grad(
            outputs=p_pa,
            inputs=z,
            grad_outputs=torch.ones_like(p_pa),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )

        # Calculate MSE loss
        return torch.mean((dP_dz - expected_dP_dz)**2)


class INRLoggerCallback(pl.Callback):
    def __init__(self, monitor_metrics, mode="min", save_path="checkpoints"):

        super().__init__()
        #debug_print()
        self.monitor_metrics = monitor_metrics
        self.mode = mode
        self.save_path = save_path
        self.best_metrics = {metric: float("inf") if mode == "min" else float("-inf") for metric in monitor_metrics}

        os.makedirs(save_path, exist_ok=True)

        # Track time
        self.total_start_time = None
        self.total_points= 0

    def on_train_start(self, trainer, pl_module):
        """record start time"""
        #debug_print()
        self.total_start_time = time.time()
        print("Training Started...")

    def on_train_epoch_start(self, trainer, pl_module):
        """Records the epoch logs it."""
        #debug_print()
        self.total_points = 0
        epoch = trainer.current_epoch
        print(f"Starting Epoch {epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        #debug_print()

        # Count points processed in this batch
        batch_size = batch["inputs"].shape[0]  # Assuming batch["inputs"] contains spatial points
        self.total_points += batch_size
        current_metrics = trainer.callback_metrics
        for metric in self.monitor_metrics:
            if metric in current_metrics:
                metric_value = current_metrics[metric].item()
                if (self.mode == "min" and metric_value < self.best_metrics[metric]) or \
                        (self.mode == "max" and metric_value > self.best_metrics[metric]):
                    self.best_metrics[metric] = metric_value
                    checkpoint_path = os.path.join(self.save_path, f"best_{metric}.ckpt")
                    trainer.save_checkpoint(checkpoint_path)
                    #print(f"New best {metric}: {metric_value:.6f} at batch {batch_idx}. Saved to {checkpoint_path}")

                wandb.log({metric: metric_value})

    def on_train_epoch_end(self, trainer, pl_module):
        """Prints, logs, and calculates epoch time."""
        #debug_print()
        epoch = trainer.current_epoch
        print(f"Finished Epoch {epoch}. Total points processed: {self.total_points}")

        wandb.log({"total_points_epoch": self.total_points})

    def on_train_end(self, trainer, pl_module):
        """Calculates and logs total training time."""
        #debug_print()
        total_duration = time.time() - self.total_start_time
        print(f"\n Training Completed! Total Time: {total_duration:.2f} sec")
        wandb.log({"total_training_time": total_duration})






