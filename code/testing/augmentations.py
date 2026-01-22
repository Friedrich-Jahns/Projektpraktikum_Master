import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import h5py
import numpy as np
from  scipy.signal import convolve2d
cwd = os.getcwd()

raw_path = Path(cwd).parent.parent/'data/PE-2024-01126-M_00_s0021_PM_Complete_Transmittance_Stitched_Flat_v004.h5'
print(raw_path)


class augmentation:
    def __init__(self,arr,mask):
        self.raw = arr
        self.mask = mask
        #plt.hist(mask)
        #plt.show()
        self.mask_gray = np.where(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)>0,1,0)
        self.raw_gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
 
    def conv_filter_1(self):
        dat = self.raw_gray
        mask = self.mask_gray.astype(bool)
        
        new = np.zeros(dat.shape)
        kernel = np.ones((3,3))/9
        dat_copy = np.copy(dat)

        for i in range(5):
            dat_copy = convolve2d(dat_copy,kernel,mode='same')
            if i %100 == 0:
                print(dat_copy.shape) 
        conv = dat_copy


        new = np.copy(dat)

        new[mask] = conv[mask]


        return new,conv

    
    def conv_filter_2(self):
        dat = self.raw_gray
        mask = self.mask_gray.astype(bool)
        new = np.copy(dat)
        conv = np.copy(dat)
        d = 2
        for l in range(1):
            for i in range(dat.shape[0]):
                for j in range(dat.shape[1]):
                    neigh = dat[i-d:i+d+1, j-d:j+d+1].flatten()
                    if not np.isclose(dat[i,j],np.median(neigh),5e-1):
                        conv[i,j] = np.median(neigh)
        new[mask] = conv[mask]
        return new, conv
def load_array_from_h5(path, bounds="0 -1 0 -1", return_size=False):

    try:
        bounds = np.array(bounds.split(" ")).astype(float).astype(int)
    except:
        print("Boundaries not in the right format")
    with h5py.File(path, "r") as f:
        Key = "Image" if "Image" in f.keys() else "exported_data"
        shape = f[Key].shape

        if return_size:
            return shape


        if False or all(
            [bounds[i] == [0, -1, 0, -1][i] for i in range(len(bounds))]
        ):
            bounds = [0, shape[0], 0, shape[1]]
        # Mask Dimensions don't match Image Dimensions
        if len(shape) <= 2:
            img = np.zeros((bounds[1] - bounds[0], bounds[3] - bounds[2]))
        else:
            img = np.zeros(
                (bounds[1] - bounds[0], bounds[3] - bounds[2], int(shape[2]))
            )

        f[Key].read_direct(
            img, (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
        )

    return img

plt.figure(figsize=(14,7))
plt.subplot(2,3,1)
plt.imshow(load_array_from_h5(raw_path,bounds=[1200,1456,1200,1456]),cmap='gray')
plt.title('uncolored raw')
plt.subplot(2,3,2)
colored = plt.imread(Path(cwd).parent.parent/'data/training/train/img/PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000_256_3456.png')
colored_mask = plt.imread(Path(cwd).parent.parent/'data/training/train/mask/PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000_256_3456.png')


obj = augmentation(colored,colored_mask)
plt.imshow(obj.raw_gray,cmap='gray')
plt.title('colored raw')
plt.subplot(2,3,3)
plt.imshow(obj.mask*obj.raw,cmap='gray')
plt.title('colored*mask')
plt.subplot(2,3,4)
plt.imshow(obj.conv_filter_1()[0],cmap='gray')
plt.title('colored conv1 filter')

plt.subplot(2,3,5)
plt.imshow(obj.conv_filter_2()[0],cmap='gray')

plt.subplot(2,3,6)
plt.imshow(obj.conv_filter_2()[1],cmap='gray')




plt.show()





#example_colored_path = 
#example_raw_path = 
