import wandb
import random
import numpy as np
import torch
import inspect

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


def debug_print():
    """Prints the current function name and class (if available) for debugging."""
    frame = inspect.currentframe().f_back  # Get the caller's frame
    function_name = frame.f_code.co_name  # Function name

    # Get class name if inside a class method
    class_name = None
    if 'self' in frame.f_locals:
        class_name = frame.f_locals['self'].__class__.__name__

    if class_name:
        print(f"Debug: {class_name}.{function_name}() is running...")
    else:
        print(f"Debug: {function_name}() is running...")

