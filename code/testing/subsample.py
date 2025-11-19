import numpy as np 
import matplotlib.pyplot as plt
import h5py
import napari
from pathlib import Path
import cv2


def load_array_from_h5(path):
     with h5py.File(path,'r') as file:
        data = file['Image'][:]
        return data





dat_path = Path.cwd().parent.parent/'data'

for i,path in enumerate(Path.iterdir(dat_path)):
    if i.suffix == '.h5':
        plt.imshow(load_array_from_h5(i))
        plt.show() 



