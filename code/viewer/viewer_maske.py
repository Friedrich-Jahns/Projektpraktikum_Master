import napari
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import os

path = Path('PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000-Image_Probabilities_1.h5')

Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
fullpath = Filepath.parent.parent/f'data/{path}'
print(fullpath)
viewer = napari.Viewer()

with h5py.File(fullpath,'r') as file:
    data = np.array(file['exported_data'][:])
#    img = np.zeros(data.shape) 
#    file['Image'].read_direct(img,(np.s_[0:-1, 0:-1], np.s_[0:-1, 0:-1])) 




viewer.add_image(data[:,:,0], name="Maske 1")
viewer.add_image(data[:,:,1], name="Maske 2")
viewer.add_image(data[:,:,2], name="Maske 3")


data_scaled = (data * 255).astype(np.uint8)
viewer.add_image(data_scaled[:,:,0], name="Maske 1_scaled")
viewer.add_image(data_scaled[:,:,1], name="Maske 2_scaled")
viewer.add_image(data_scaled[:,:,2], name="Maske 3_scaled")
napari.run()
