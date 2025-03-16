import pytorch_lightning as pl
import torch
from torch import nn
from utils import instantiate_from_config
import time
import wandb
import os
from helpers import set_seed, debug_print




class INRModel(pl.LightningModule):
    def __init__(self, config):
        """
    Base PINN model that instantiates the INR network dynamically from config,
    and includes PDE-based losses (hydrostatic loss and hypsometric regularizer).
    """
        super().__init__()
        debug_print()
        self.config = config.model
        self.net = instantiate_from_config(self.config)

        # Loss hyper params\
        self.physics_weight = self.config.loss_config.physics_weight
        self.regularizer_weight = self.config.loss_config.regularizer_weight

        # Dynamically configure the loss function from config.
        # If no loss is provided, default to MSELoss.
        loss_cfg = self.config.get("loss_config", {})
        loss_type = loss_cfg.get("type", "MSELoss")
        loss_params = loss_cfg.get("params", {})
        loss_class = getattr(nn, loss_type, nn.MSELoss)
        self.loss = loss_class(**loss_params)

        self.physics_loss = HydrostaticLoss()  # Instantiate physics loss module
        self.gh_min = -428.6875
        self.gh_max = 48664.082

        # these can be commented out if not need for your task
        self.GRAVITY = 9.80665 # gravitational acceleration
        self.Rd = 287 # dry air constant


    def forward(self, x):
        debug_print()
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # Keep inputs as float32 but enable gradients
        inputs = batch['inputs'].requires_grad_(True)  # [norm_gh, x, y, z]
        target = batch['target']
        pde_inputs = batch['pde_inputs']

        # Forward pass (predicts normalized temperature)
        pred_norm = self.forward(inputs)  # Uses normalized inputs

        # Data loss (on normalized values)
        data_loss = self.loss(pred_norm, target)

        # Convert to REAL VALUES with gradient tracking
        #K_min, K_max = 183, 330

        # Denormalize while preserving gradients
        #T_real_pred = (pred_norm + 1) / 2 * (K_max - K_min) + K_min
        #gh_real = (inputs[:, 0] + 1) / 2 * (self.gh_max - self.gh_min) + self.gh_min  # [N,]

        # Compute physics loss using REAL VALUES with gradient tracking
        #physics_loss = self.physics_loss(T_real_pred, gh_real, pde_inputs)
        #physics_regulariser = self.compute_physics_reg(T_real_pred, gh_real, pde_inputs)

        total_loss = data_loss #+ self.physics_weight*physics_loss + self.regularizer_weight*physics_regulariser

        # logging into wandb
        self.log("train/data_loss", data_loss, on_step=True, on_epoch=True)
        #self.log("train/physics_loss", physics_loss, on_step=True, on_epoch=True)
        #self.log("train/physics_regulariser", physics_regulariser, on_step=True, on_epoch=True)
        #self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        debug_print()
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

    # def compute_physics_loss(self, T_real_pred, gh_real, pde_inputs):
    #     """Physics loss using real values with proper gradient connection"""
    #     p = pde_inputs["pressure_level"]  # [N,1]
    #     q = pde_inputs["specific_humidity"]  # [N,1]
    #
    #     # Virtual temperature
    #     T_v = T_real_pred * (1 + 0.61 * q)  # [N,1]
    #
    #     # Hydrostatic equation components
    #     p_pa = p * 100  # Convert hPa to Pa
    #     expected_dT_dz = - (p_pa * self.GRAVITY) / (self.Rd * T_v)  # [N,1]
    #
    #     # Compute gradient of temperature w.r.t REAL geopotential height
    #     dT = torch.autograd.grad(
    #         outputs=T_real_pred,
    #         inputs=gh_real,  # Directly use real gh values
    #         grad_outputs=torch.ones_like(T_real_pred),
    #         create_graph=True,  # Needed for higher-order gradients
    #         retain_graph=True,
    #         allow_unused=False
    #     )[0].unsqueeze(-1)  # [N,1]
    #
    #     # Compute MSE between predicted and expected gradient
    #     hydrostatic_loss = torch.mean((dT - expected_dT_dz) ** 2)
    #     return hydrostatic_loss

    def compute_physics_reg(self, real_pred, gh_real, pde_inputs):
        """ A loss function for designing a physics regularization term, by default the loss is the hypsometric function


        For each spatial point, the expected thickness between 1000 mb and the current pressure level p
        is computed as:

             Δz_expected = (Rd/g) * T_real_pred * ln(1000 / p)

        The actual thickness is:

             Δz_actual = base_geopotential_height - geopotential_height
        """

        debug_print()
        # Extract PDE variables
        p = pde_inputs["pressure_level"]    # shape (N, 1), in hPa
        gh_base = pde_inputs["base_geopotential_height"]  # shape (N, 1), in meters

        # Compute expected thickness using the hypsometric equation.
        expected_thickness = (self.Rd / self.GRAVITY) * real_pred * torch.log(1000.0 / p)

        # Compute actual thickness: difference between the base geopotential height and the current geopotential height.
        actual_thickness = gh_base - gh_real

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
        """
        Args:
            T_real: (N,1) Predicted temperature in Kelvin
            gh_real: (N,) Geopotential height in meters (MUST be float32 and require grad)
            pde_inputs: Dict containing:
                - pressure_level: (N,1) Pressure in hPa
                - specific_humidity: (N,1) kg/kg
        """
        # Virtual temperature calculation
        q = pde_inputs["specific_humidity"]
        T_v = T_real * (1 + 0.61 * q)  # (N,1)

        # Hydrostatic balance equation components
        p_pa = pde_inputs["pressure_level"] * self.hpa_to_pa  # (N,1)
        expected_dT_dz = - (p_pa * self.gravity) / (self.Rd * T_v)  # (N,1)

        # Compute gradient dT/dz (requires gh_real to have requires_grad=True)
        gh_real = gh_real.unsqueeze(-1)  # (N,1)
        dT = torch.autograd.grad(
            outputs=T_real,
            inputs=gh_real,
            grad_outputs=torch.ones_like(T_real),
            create_graph=True,  # Critical for backprop
            retain_graph=True,
            allow_unused=False
        )[0]  # (N,1)

        # Calculate MSE loss
        return torch.mean((dT - expected_dT_dz)**2)


class INRLoggerCallback(pl.Callback):
    def __init__(self, monitor_metrics, mode="min", save_path="checkpoints"):

        super().__init__()
        debug_print()
        self.monitor_metrics = monitor_metrics
        self.mode = mode
        self.save_path = save_path
        self.best_metrics = {metric: float("inf") if mode == "min" else float("-inf") for metric in monitor_metrics}

        os.makedirs(save_path, exist_ok=True)

        # Track time
        self.total_start_time = None

    def on_train_start(self, trainer, pl_module):
        """record start time"""
        debug_print()
        self.total_start_time = time.time()
        print("Training Started...")

    def on_train_epoch_start(self, trainer, pl_module):
        """Records the epoch logs it."""
        debug_print()
        epoch = trainer.current_epoch
        print(f"Starting Epoch {epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        debug_print()
        current_metrics = trainer.callback_metrics
        for metric in self.monitor_metrics:
            if metric in current_metrics:
                metric_value = current_metrics[metric].item()
                if (self.mode == "min" and metric_value < self.best_metrics[metric]) or \
                        (self.mode == "max" and metric_value > self.best_metrics[metric]):
                    self.best_metrics[metric] = metric_value
                    checkpoint_path = os.path.join(self.save_path, f"best_{metric}.ckpt")
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"New best {metric}: {metric_value:.6f} at batch {batch_idx}. Saved to {checkpoint_path}")

                wandb.log({metric: metric_value})

    def on_train_epoch_end(self, trainer, pl_module):
        """Prints, logs, and calculates epoch time."""
        debug_print()
        epoch = trainer.current_epoch
        print(f"Finished Epoch {epoch}")

    def on_train_end(self, trainer, pl_module):
        """Calculates and logs total training time."""
        debug_print()
        total_duration = time.time() - self.total_start_time
        print(f"\n Training Completed! Total Time: {total_duration:.2f} sec")
        wandb.log({"total_training_time": total_duration})






