import numpy as np 
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
import os

def load_array_from_h5(path):
     with h5py.File(path,'r') as file:
        Key = "Image" if "Image" in file.keys() else "exported_data"
        data = file[Key][:]
        return np.array(data)


Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = Filepath.parent.parent/'data'
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

for i in tqdm(range(0,img.shape[0],250)): # Je 500 für keinen overlap, 250 -> jeder bereich des bildes ist in 4 ausschnitten enthalten
    for j in range(0,img.shape[1],250):   # " ^ " 
        sub_img = img[i:i+500, j:j+500] 

        plt.imsave(save_path/f'{i}_{j}.png',sub_img,cmap="gray")