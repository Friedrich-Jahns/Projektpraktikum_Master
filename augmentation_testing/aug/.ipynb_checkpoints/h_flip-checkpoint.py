import torch
import torchvision.transforms.functional as TF
import random

def get_augmentation():
    def augment(img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask
    return augment