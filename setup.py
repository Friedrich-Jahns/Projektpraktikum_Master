from pathlib import Path
import os

folders = [
    "data/models",
    "data/training/train/img",
    "data/training/train/mask",
    "data/training/val/img",
    "data/training/val/mask",
    'data/ilastik',
    'data/raw/img',
    'data/raw/vessel_mask',
    'data/raw/bgr_mask'


]

dat_path = Path(os.path.dirname(os.path.abspath(__file__)))/'data'
target_path = Path(os.path.dirname(os.path.abspath(__file__)))/'data'/'raw'


for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
print(dat_path)

if True:
    for j,path in enumerate(Path.iterdir(dat_path)):
        if any([i=='v000' or i=='v000-Image' for i in str(path.stem).split('_')]):
            if path.suffix == '.h5':
                if any([i=='Probabilities' for i in str(path.stem).split('_')]):
                    type = 'vessel_mask'
                    if any([i=='background' for i in str(path.stem).split('_')]):
                        type = 'bgr_mask'
                else:
                    type = 'img'
                os.rename(path,target_path/type/path.name)
