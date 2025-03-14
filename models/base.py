import pytorch_lightning as pl
import torch
from torch import nn
from utils import instantiate_from_config




class INRModel(pl.LightningModule):
    def __init__(self, config):
        """
    Base PINN model that instantiates the INR network dynamically from config,
    and includes PDE-based losses (hydrostatic loss and hypsometric regularizer).
    """
        super().__init__()
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

        # these can be commented out if not need for your task
        self.GRAVITY = 9.80665 # gravitational acceleration
        self.Rd = 287 # dry air constant


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        # inputs for foward pass
        inputs = batch['inputs']  # shape (N, 4) - N, [pressure_level_norm, x_coord, y_coord, z_coord]
        target = batch['target'] # shape (N, 1) - N, [temperature_norm]
        pde_inputs = batch['pde_inputs'] # shape (N, 3) - N, [pressure_level, geopotential_height, specific_humidity]

        pred_norm = self.forward(inputs)

        # loss calculation
        data_loss = self.loss(pred_norm, target)

        K_min, K_max = 183, 330
        T_real_pred = (pred_norm + 1) / 2 * (K_max - K_min) + K_min  # shape: (N, 1)

        physics_loss = self.compute_physics_loss(T_real_pred, pde_inputs)
        physics_regulariser = self.compute_physics_reg(T_real_pred, pde_inputs)

        total_loss = data_loss + self.physics_weight*physics_loss + self.regularizer_weight*physics_regulariser

        # logging into wandb
        self.log("train/data_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train/physics_loss", physics_loss, on_step=True, on_epoch=True)
        self.log("train/physics_regulariser", physics_regulariser, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):

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


    def compute_physics_loss(self, real_pred, pde_inputs):
        """ A loss function for designing a physics loss, by default the loss is the hydrostatic function
        change the logic of this function to suit your needs"""


        p = pde_inputs["pressure_level"]
        gh = pde_inputs["geopotential_height"].clone().requires_grad_(True)
        q = pde_inputs["specific_humidity"]

        # compute the virtual temperature
        T_v = real_pred * (1 + 0.61 * q)

        # convert pressure to Pa from hPa
        p_pa = p * 100
        expected_dT_dz = - (p_pa * self.GRAVITY) / (self.Rd * T_v)

        # Use autograd to compute dT/dz with respect to geopotential height
        dT = torch.autograd.grad(
            outputs=real_pred,
            inputs=gh,
            grad_outputs=torch.ones_like(real_pred),
            create_graph=True,
            retain_graph=True)[0]  # shape (N, 1)

        hydrostatic_loss = torch.mean((dT - expected_dT_dz) ** 2)

        return hydrostatic_loss


    def compute_physics_reg(self, real_pred, pde_inputs):
        """ A loss function for designing a physics regularization term, by default the loss is the hypsometric function


        For each spatial point, the expected thickness between 1000 mb and the current pressure level p
        is computed as:

             Δz_expected = (Rd/g) * T_real_pred * ln(1000 / p)

        The actual thickness is:

             Δz_actual = base_geopotential_height - geopotential_height
        """
        # Extract PDE variables
        p = pde_inputs["pressure_level"]    # shape (N, 1), in hPa
        gh = pde_inputs["geopotential_height"]  # shape (N, 1), in meters
        gh_base = pde_inputs["base_geopotential_height"]  # shape (N, 1), in meters

        # Compute expected thickness using the hypsometric equation.
        expected_thickness = (self.Rd / self.GRAVITY) * real_pred * torch.log(1000.0 / p)

        # Compute actual thickness: difference between the base geopotential height and the current geopotential height.
        actual_thickness = gh_base - gh

        # Compute the hypsometric loss as the mean squared error between expected and actual thickness.
        hypsometric_loss = torch.mean((actual_thickness - expected_thickness) ** 2)

        return hypsometric_loss







