import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random

import torch
# from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
# import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

import os
from pathlib import Path

# from sklearn.metrics import f1_score

dat_path = Path.cwd().parent.parent/'data'/'training'


path = {
    "train": dat_path/'train',
    "val": dat_path/'val',
    "test": dat_path/'test',
    # "submission": dat_path,
}


##########
# Functions
##########
def rgb_to_grayscale(image: torch.Tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device).view(
        1, 3, 1, 1
    )
    grayscale = (image * weights).sum(dim=1, keepdim=True)
    return grayscale


def grayscale_to_rgb(grayscale: torch.Tensor):
    return grayscale.expand(-1, 3, -1, -1)


def filter_1(x: torch.Tensor):  # Threshold
    arr = torch.where(
        x > torch.mean(x), torch.tensor(0, dtype=x.dtype, device=x.device), x
    )
    return arr


def filter_2(x: torch.Tensor):  # Threshold and Highpass
    image = torch.where(
        x > torch.mean(x), torch.tensor(0, dtype=x.dtype, device=x.device), x
    )
    image = rgb_to_grayscale(image)
    laplacian_kernel = torch.tensor(
        [[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32
    )
    high_pass = F.conv2d(image, laplacian_kernel, padding=1)
    high_pass = grayscale_to_rgb(high_pass)
    return high_pass

class CreateDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img = F.to_tensor(img)
            mask = np.array(mask)  # shape H x W x 3 falls RGB
            if mask.ndim == 3:    # RGB → 1 Kanal
                # Beispiel: nur roter Kanal als Klassen-Index (0 oder 1)
                mask = mask[:,:,0]
                mask = (mask > 128).astype(np.int64)  # Hintergrund=0, Objekt=1

            mask = torch.as_tensor(mask, dtype=torch.long)  # shape H x W

        return img, mask




###########
# Randomseed
###########
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


##############
# Augmentierung
##############
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Lambda(lambda img: filter_1(img)),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
    ]
)
###########
# Dataloader
###########
img_paths   = [path['train']/'img'/f for f in os.listdir(path['train']/'img')]
mask_paths  = [path['train']/'mask'/f for f in os.listdir(path['train']/'mask')]
# print(img_paths)

dataset = CreateDataset(img_paths,mask_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#################
# Pretrained model
#################
model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)

# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 3)

##############
# Loss function
##############
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# loss_fn = nn.CrossEntropyLoss()


#########
#Training
#########
for epoch in range(10):
    for imgs, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs)['out']  # Shape: [B, C, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

###########
#Validation
###########

# valid_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda img: filter_1(img)),
#     transforms.Resize((224, 224)),
# ])


# valid_dataset = CreateDataset(path['val']/'img',path['val']/'mask')
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # model.to(device)
# model.eval()

# all_labels = []
# all_preds = []

# with torch.no_grad():
#     for inputs, labels in valid_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         outputs = model(inputs)
        
#         _, predicted = torch.max(outputs, 1)

#         all_labels.extend(labels.cpu().numpy())
#         all_preds.extend(predicted.cpu().numpy())

# f1 = f1_score(all_labels, all_preds, average='weighted')  # Für unbalancierte Klassen ist 'weighted' sinnvoll
# print(f'F1-Score: {f1:.4f}')