import importlib

def instantiate_from_config(config_dict):
    """
    Instantiate a class from a config dictionary that contains a _target_ key.
    All remaining keys in the dictionary are passed as keyword arguments to the class.
    """
    target_str = config_dict.pop("_target_")
    module_path, class_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config_dict)