import napari
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


cwd = Path.cwd()
path = Path('PE-2024-01126-M_00_s0021_PM_Complete_Transmittance_Stitched_Flat_v004.h5')
fullpath = Path.cwd().parent.parent/f'data/{path}'
print(fullpath)
viewer = napari.Viewer()

with h5py.File(fullpath,'r') as file:
    data = file['Image'][:]
#    img = np.zeros(data.shape) 
#    file['Image'].read_direct(img,(np.s_[0:-1, 0:-1], np.s_[0:-1, 0:-1])) 




viewer.add_image(data, name="Mein H5-Bild")
napari.run()
