import numpy as np 
import matplotlib.pyplot as plt
import h5py
import napari
from pathlib import Path
import cv2
import napari
import os

def load_array_from_h5(path):
     with h5py.File(path,'r') as file:
        data = file['Image'][:]
        return data

def binary(img):
    img = np.where(img>.3,0,1)
    return img


viewer = napari.Viewer()

Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = Filepath.parent.parent/'data'

for i,path in enumerate(Path.iterdir(dat_path)):
    if path.suffix == '.h5':
        img = load_array_from_h5(path)
        img = cv2.normalize(img,None,norm_type=cv2.NORM_MINMAX)
        img = np.where(img>.6,0,img)
        #plt.hist(img.flatten())
        #plt.show()
        img = cv2.normalize(img,None,norm_type=cv2.NORM_MINMAX)
        #img = binary(img)
        viewer.add_image(img,name=f'img_nr:{i}')


napari.run()
