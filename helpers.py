import wandb
import random
import numpy as np
import torch
import inspect
import os
import datetime


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
    """Logs the current function name and class (if available) to a timestamped debug file."""

    # Generate log filename once per execution
    if not hasattr(debug_print, "log_filename"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_print.log_filename = f"debug_log_{timestamp}.txt"

    # Get caller details
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name

    # Check if inside a class method
    class_name = frame.f_locals['self'].__class__.__name__ if 'self' in frame.f_locals else None

    # Create log entry
    log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
    log_entry += f"{class_name + '.' if class_name else ''}{function_name}() is running...\n"

    # Append to the log file
    with open(debug_print.log_filename, "a") as log_file:
        log_file.write(log_entry)





