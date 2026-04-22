import importlib

def load_augmentation(name, ref_dir=None):
    module = importlib.import_module(f"aug.{name}")
    return module.get_augmentation(ref_dir=ref_dir)