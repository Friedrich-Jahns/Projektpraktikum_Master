import numpy as np 
import matplotlib.pyplot as plt
import h5py
import napari
from pathlib import Path
import cv2
import napari


def load_array_from_h5(path):
     with h5py.File(path,'r') as file:
        data = file['Image'][:]
        return data



dat_path = Path.cwd().parent.parent/'data'
paths={}
for i,path in enumerate(Path.iterdir(dat_path)):
        if path.suffix == '.h5':
                print(f'{i} : {path}')
                paths[i] = path
# print(paths[1])

key = input('Dat. Nr.')
img = load_array_from_h5(Path(paths[int(key)]))


