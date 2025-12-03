import numpy as np 
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


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

save_path = dat_path/f'full_size_png_{file_path.stem}'
save_path.mkdir(parents=True, exist_ok=True)


plt.imsave(save_path/f'file.png',img,cmap="gray")