import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
from pathlib import Path
from PIL import Image

def load_ref_tiles(ref_dir):
    """Alle ungefärbten Tiles einladen als Referenzpool"""
    paths = list(Path(ref_dir).glob('*.png'))
    tiles = [torch.tensor(
        np.array(Image.open(p).convert('L')), dtype=torch.float32
    ).unsqueeze(0) / 255.0 for p in paths]
    return tiles

def histogram_match(img, ref):
    src = (img.squeeze().numpy() * 255).astype(np.uint8)
    ref = (ref.squeeze().numpy() * 255).astype(np.uint8)
    src_cdf = np.histogram(src.flatten(), 256, [0,256])[0].cumsum()
    ref_cdf = np.histogram(ref.flatten(), 256, [0,256])[0].cumsum()
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]: j += 1
        lut[i] = j
    return torch.tensor(lut[src], dtype=torch.float32).unsqueeze(0) / 255.0

def blur_masked(img, mask, sigma):
    arr     = (img.squeeze().numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    m       = mask.squeeze().numpy()
    result  = arr * (1 - m) + blurred * m
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0) / 255.0

def get_augmentation(ref_dir):
    ref_tiles = load_ref_tiles(ref_dir)

    def augment(img, mask):
        # Geometrie – auf beide anwenden
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # Histogram Matching gegen zufälliges ungefärbtes Tile
        ref = random.choice(ref_tiles)
        img = histogram_match(img, ref)

        # Maskierter Blur
        img = blur_masked(img, mask, sigma=random.uniform(0.5, 1.0))

        return img.clamp(0, 1), mask

    return augment