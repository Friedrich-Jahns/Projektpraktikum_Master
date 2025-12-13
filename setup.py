from pathlib import Path

folders = [
    "data/models",
    "data/training/train/img",
    "data/training/train/mask",
    "data/training/val/img",
    "data/training/val/mask",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)