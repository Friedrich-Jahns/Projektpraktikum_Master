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




###########
# Validation
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
