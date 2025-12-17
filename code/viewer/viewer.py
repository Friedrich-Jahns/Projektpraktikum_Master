import napari
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import os

cwd = Path.cwd()
path = Path('/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/raw/img/v004_3.h5')

Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
fullpath = Filepath.parent.parent/f'data/{path}'
print(fullpath)
viewer = napari.Viewer()

with h5py.File(fullpath,'r') as file:
    data = file['Image'][:]
#    img = np.zeros(data.shape) 
#    file['Image'].read_direct(img,(np.s_[0:-1, 0:-1], np.s_[0:-1, 0:-1])) 




viewer.add_image(data, name="Mein H5-Bild")

napari.run()
