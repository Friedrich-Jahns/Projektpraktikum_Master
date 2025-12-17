import napari
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import os
from tqdm import tqdm
import cc3d
from scipy.ndimage import label

def load_array_from_h5(path):
     with h5py.File(path,'r') as file:
        Key = "Image" if "Image" in file.keys() else "exported_data"
        data = file[Key][:]
        return np.array(data)


def create_background_mask(background_mask_initial, min_size=500):
    dat = (background_mask_initial == 0).astype(np.uint8)
    labeled, num_labels = label(dat, structure=np.ones((3,3)))

    counts = np.bincount(labeled.ravel())
    mask = counts[labeled] >= min_size
    return mask.astype(np.uint8)




Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = Filepath.parent.parent/'data'/'raw'

paths={}
for i,folder in enumerate(Path.iterdir(dat_path)):
    for j,path in enumerate(Path.iterdir(folder)):
        if path.suffix == '.h5':
                print(f'{i} : {path}')
        img = load_array_from_h5(path)




#save_path = dat_path / 'training' / 'train' / 'img'/ f'subsample_{file_path.stem}'
#save_path.mkdir(parents=True, exist_ok=True)

#for i in tqdm(range(0,img.shape[0],250)): # Je 500 für keinen overlap, 250 -> jeder bereich des bildes ist in 4 ausschnitt>
#    for j in range(0,img.shape[1],250):   # " ^ "
#        sub_img = img[i:i+500, j:j+500]

#        plt.imsave(save_path/f'{i}_{j}.png',sub_img,cmap="gray")


#img_nr = 2
#plt.figure(figsize=(15,10))
#plt.subplot(121)
#plt.imshow(data[:,:,img_nr],cmap='gray')
#plt.subplot(122)
#plt.imshow(create_background_mask(data[:,:,img_nr]),cmap='gray')

#plt.show()
