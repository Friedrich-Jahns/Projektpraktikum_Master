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
        return np.array(data)



dat_path = Path.cwd().parent.parent/'data'
paths={}
for i,path in enumerate(Path.iterdir(dat_path)):
        if path.suffix == '.h5':
                print(f'{i} : {path}')
                paths[i] = path
# print(paths[1])

key = input('Dat. Nr.')
file_path = Path(paths[int(key)])
img = load_array_from_h5(file_path)

save_path = dat_path/f'subsample_{file_path.stem}'
save_path.mkdir(parents=True, exist_ok=True)

for i in range(0,img.shape[0],500):
    for j in range(0,img.shape[1],500):    
        sub_img = img[i:i+500, j:j+500] 

        plt.imsave(save_path/f'{i}_{j}.png',sub_img,cmap="gray")