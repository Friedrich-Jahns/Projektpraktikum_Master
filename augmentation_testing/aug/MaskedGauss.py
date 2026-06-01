import torch
import cv2
import random
import numpy as np

def blur_masked_old(img, mask, sigma):
    arr     = (img.squeeze().numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    m       = mask.squeeze().numpy()
    result  = arr * (1 - m) + blurred * m
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0) / 255.0

def blur_masked(img, mask, sigma):
    arr     = (img.squeeze().numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    m       = np.clip(mask.squeeze().numpy(), 0, 1)
    result  = arr * (1 - m) + blurred * m
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0) / 255.0

def get_augmentation(ref_dir=None):
    def augment(img, mask):
        # Maskierter Blur
        img = blur_masked(img, mask, sigma=random.uniform(0.5, 1.0))
        # clamp schneidet die values bei 0 und 1 jeweils ab: (1.5, -10, 0.3) -> (1, 0, 0.3)
        return img.clamp(0, 1), mask
    return augment