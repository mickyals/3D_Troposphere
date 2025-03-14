import wandb
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_count():
    return torch.cuda.device_count()

def get_logger():
    pass

def get_config():
    pass

def get_model():
    pass

def get_trainer():
    pass

def end_timer():
    pass

def track_lr():
    pass

def init_weights(m):
    pass

def init_wandb(api_key=None, project="3D_Atmosphere", entity=None, config=None):
    wandb.login(key=api_key)
    wandb.init(project=project, entity=entity, config=config)

def init_logger():
    pass