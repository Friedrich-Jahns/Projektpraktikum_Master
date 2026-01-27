from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np

class img_dataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,i):
        img_path = os.path.join(self.img_dir,self.images[i])
        mask_path = os.path.join(self.mask_dir,self.masks[i])

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0

        mask = (mask>.5).float()

        if self.transform:
            img,mask = self.transform(img,mask)
            
        return img, mask


def dataloader(img_dir,mask_dir,bs=4,shuffle=True,num_workers=0):
    dataset = img_dataset(img_dir,mask_dir)
    dataloader = DataLoader(
    dataset,
    batch_size = bs,
    shuffle = shuffle,
    num_workers = num_workers
    )
    return dataloader
    
