import importlib

def load_augmentation(name):
    module = importlib.import_module(f"aug.{name}")
    return module.get_augmentation()