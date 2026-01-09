import numpy as np 
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
import os

def load_array_from_h5(path,bounds):
    try:
        bounds = np.array(bounds.split(' ')).astype(float).astype(int)
    except:
        print('Boundaries not in the right format')
    with h5py.File(path,'r') as f:
        Key = "Image" if "Image" in f.keys() else "exported_data"
        shape = f[Key].shape
        
        #Show Full Image
        if bounds == '0 -1 0 -1' or all([bounds[i]==[0,-1,0,-1][i] for i in range(len(bounds))]):
            bounds = [0,shape[0],0,shape[1]] 
        #Mask Dimensions don't match Image Dimensions
        if len(shape)<=2:
            img = np.zeros((bounds[1]-bounds[0],bounds[3]-bounds[2]))
        else:
            img = np.zeros((bounds[1]-bounds[0],bounds[3]-bounds[2],int(shape[2])))

        f[Key].read_direct(img,(slice(bounds[0],bounds[1]),slice(bounds[2],bounds[3])))

    return img

def get_largest_component(data):
    print('filer')
    






class mask_layers:
    def __init__(self,image_path,bounds):
        self.img_path = image_path
        self.vessel_path = Path(image_path).with_name(Path(image_path).stem + '-Image_Probabilities_255_8b' + Path(image_path).suffix)
        self.bckg_path = Path(image_path).with_name(Path(image_path).stem + '-Image_Probabilities_background' + Path(image_path).suffix)
        self.bounds = bounds

    def image_data(self):
        dat = load_array_from_h5(self.img_path,self.bounds)
        return dat

    def background_data(self):
        dat = load_array_from_h5(self.bckg_path,self.bounds)[:,:,1]
        return dat

    def vessel_data(self):
        dat = load_array_from_h5(self.vessel_path,self.bounds)[:,:,0]
        return dat


    def background_data_threshed(self):
        unthreshed  = self.background_data()
        dat = np.where(unthreshed>= 100,1,0)
        return dat

    def vessel_data_threshed(self):
        unthreshed  = self.vessel_data()
        dat = np.where(unthreshed< 75,0,1)
        return dat

    def vessel_data_threshed_nobg(self):
        dat = np.multiply(self.vessel_data_threshed(),self.background_data_threshed())
        return dat
    
    def cc_filter(self):
        input = self.vessel_data_threshed_nobg()
        analysis = cv2.connectedComponentsWithStats(input, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis
        
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
        


cwd = Path(os.path.dirname(os.path.abspath(__file__)))
dat_path = cwd.parent.parent/'data'
path_list = [
    dat_path / 'PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000.h5'
    ]

#Demo
if True:
    #obj = mask_layers(path_list[0],'2200 3200 5000 6000')
    obj = mask_layers(path_list[0],'0 -1 0 -1')
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(obj.image_data(),cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(obj.background_data_threshed(),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(obj.vessel_data_threshed(),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(obj.vessel_data_threshed_nobg(),cmap='gray')
    plt.show()


        


#Filepath = Path(os.path.dirname(os.path.abspath(__file__)))
#dat_path = Filepath.parent.parent/'data'
#paths={}
#for i,path in enumerate(Path.iterdir(dat_path)):
#        if path.suffix == '.h5':
#                print(f'{i} : {path}')
#                paths[i] = path
# print(paths[1])

#key = input('Dat. Nr.')
#file_path = Path(paths[int(key)])
#img = load_array_from_h5(file_path)

#save_path = dat_path / 'training' / 'train' / 'img'/ f'subsample_{file_path.stem}'
#save_path.mkdir(parents=True, exist_ok=True)

#for i in tqdm(range(0,img.shape[0],250)): # Je 500 für keinen overlap, 250 -> jeder bereich des bildes ist in 4 ausschnitten enthalten
#    for j in range(0,img.shape[1],250):   # " ^ " 
#        sub_img = img[i:i+500, j:j+500] 

#        plt.imsave(save_path/f'{i}_{j}.png',sub_img,cmap="gray")
