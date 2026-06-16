import cv2

    
"""
joint_transforms.py
-------------------
torchvision-kompatible Transforms, die Bild UND Maske identisch transformieren.
Verwendung genau wie transforms.Compose([...]).

Beispiel:
    transform = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=30),
        JointRandomResizedCrop(size=(256, 256)),
        JointColorJitter(brightness=0.3, contrast=0.3),  # nur auf Bild
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),       # nur auf Bild
    ])

    image_t, mask_t = transform(image_pil, mask_pil)
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Basisklasse
# ---------------------------------------------------------------------------

class JointTransform:
    """Abstrakte Basisklasse.  Jede Unterklasse implementiert __call__(image, mask)."""

    def __call__(self, image: Image.Image, mask: Image.Image):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class JointCompose(JointTransform):
    """Wie transforms.Compose, aber für (image, mask)-Paare."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        lines = [f"  {t}" for t in self.transforms]
        return "JointCompose([\n" + "\n".join(lines) + "\n])"


# ---------------------------------------------------------------------------
# Geometrische Transforms  (beide identisch transformiert)
# ---------------------------------------------------------------------------

class JointMaskedGauss(JointTransform):
    def __init__(self, sigmaLow=0.5, sigmaUpper=1.0):
        self.sigmaLow = sigmaLow
        self.sigmaUpper = sigmaLow

    def __call__(self, image, mask):
        sig = random.uniform(self.sigmaLow, self.sigmaUpper)
        arr     = (image.squeeze().numpy() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(arr, (0, 0), sig)
        m       = np.clip(mask.squeeze().numpy(), 0, 1)
        result  = arr * (1 - m) + blurred * m
        return torch.tensor(result, dtype=torch.float32).unsqueeze(0) / 255.0, mask

class JointResize(JointTransform):
    """Skaliert Bild und Maske auf dieselbe Größe."""

    def __init__(self, size, interpolation_image=TF.InterpolationMode.BILINEAR,
                 interpolation_mask=TF.InterpolationMode.NEAREST):
        self.size = size  # (H, W) oder int
        self.interp_img = interpolation_image
        self.interp_mask = interpolation_mask

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=self.interp_img)
        mask  = TF.resize(mask,  self.size, interpolation=self.interp_mask)
        return image, mask


class JointRandomHorizontalFlip(JointTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask


class JointRandomVerticalFlip(JointTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        return image, mask


class JointRandomRotation(JointTransform):
    """
    Zufällige Rotation mit demselben Winkel für Bild und Maske.
    fill_mask=0 → Hintergrundklasse für neue Pixel.
    """

    def __init__(self, degrees, fill_image=0, fill_mask=0,
                 interpolation_image=TF.InterpolationMode.BILINEAR,
                 interpolation_mask=TF.InterpolationMode.NEAREST):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.fill_image = fill_image
        self.fill_mask  = fill_mask
        self.interp_img  = interpolation_image
        self.interp_mask = interpolation_mask

    def __call__(self, image, mask):
        angle = random.uniform(*self.degrees)
        image = TF.rotate(image, angle, interpolation=self.interp_img,  fill=self.fill_image)
        mask  = TF.rotate(mask,  angle, interpolation=self.interp_mask, fill=self.fill_mask)
        return image, mask


class JointRandomResizedCrop(JointTransform):
    """
    Wie RandomResizedCrop: wählt einmalig (i, j, h, w) und wendet es auf beide an.
    """

    def __init__(self, size, scale=(0.8, 1.0), ratio=(0.75, 1.333),
                 interpolation_image=TF.InterpolationMode.BILINEAR,
                 interpolation_mask=TF.InterpolationMode.NEAREST):
        self.size   = size if isinstance(size, (list, tuple)) else (size, size)
        self.scale  = scale
        self.ratio  = ratio
        self.interp_img  = interpolation_image
        self.interp_mask = interpolation_mask

    def __call__(self, image, mask):
        i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = TF.resized_crop(image, i, j, h, w, self.size, interpolation=self.interp_img)
        mask  = TF.resized_crop(mask,  i, j, h, w, self.size, interpolation=self.interp_mask)
        return image, mask


class JointRandomCrop(JointTransform):
    """Zufälliger Crop mit identischer Position."""

    def __init__(self, size, padding=None, pad_if_needed=False,
                 fill_image=0, fill_mask=0,
                 padding_mode="constant"):
        self.size           = size if isinstance(size, (list, tuple)) else (size, size)
        self.padding        = padding
        self.pad_if_needed  = pad_if_needed
        self.fill_image     = fill_image
        self.fill_mask      = fill_mask
        self.padding_mode   = padding_mode

    def __call__(self, image, mask):
        # optionales Padding
        if self.padding:
            image = TF.pad(image, self.padding, self.fill_image, self.padding_mode)
            mask  = TF.pad(mask,  self.padding, self.fill_mask,  self.padding_mode)

        if self.pad_if_needed:
            w, h = TF.get_image_size(image)
            if w < self.size[1]:
                image = TF.pad(image, (self.size[1] - w, 0), self.fill_image, self.padding_mode)
                mask  = TF.pad(mask,  (self.size[1] - w, 0), self.fill_mask,  self.padding_mode)
            if h < self.size[0]:
                image = TF.pad(image, (0, self.size[0] - h), self.fill_image, self.padding_mode)
                mask  = TF.pad(mask,  (0, self.size[0] - h), self.fill_mask,  self.padding_mode)

        i, j, h, w = T.RandomCrop.get_params(image, self.size)
        image = TF.crop(image, i, j, h, w)
        mask  = TF.crop(mask,  i, j, h, w)
        return image, mask


class JointCenterCrop(JointTransform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = TF.center_crop(image, self.size)
        mask  = TF.center_crop(mask,  self.size)
        return image, mask


class JointRandomAffine(JointTransform):
    """
    Zufällige Affin-Transformation (Translation, Scherung, Skalierung)
    mit identischen Parametern für Bild und Maske.
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None,
                 fill_image=0, fill_mask=0,
                 interpolation_image=TF.InterpolationMode.BILINEAR,
                 interpolation_mask=TF.InterpolationMode.NEAREST):
        self.degrees     = degrees
        self.translate   = translate
        self.scale       = scale
        self.shear       = shear
        self.fill_image  = fill_image
        self.fill_mask   = fill_mask
        self.interp_img  = interpolation_image
        self.interp_mask = interpolation_mask

    def __call__(self, image, mask):
        img_size = TF.get_image_size(image)
        params   = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, img_size
        )
        angle, translations, scale, shear = params
        image = TF.affine(image, angle, translations, scale, shear,
                          interpolation=self.interp_img,  fill=self.fill_image)
        mask  = TF.affine(mask,  angle, translations, scale, shear,
                          interpolation=self.interp_mask, fill=self.fill_mask)
        return image, mask


class JointElasticTransform(JointTransform):
    """
    Elastische Verformung – identische Displacement-Maps für Bild und Maske.
    Erfordert torchvision >= 0.13.
    """

    def __init__(self, alpha=50.0, sigma=5.0,
                 fill_image=0, fill_mask=0,
                 interpolation_image=TF.InterpolationMode.BILINEAR,
                 interpolation_mask=TF.InterpolationMode.NEAREST):
        self.alpha       = alpha
        self.sigma       = sigma
        self.fill_image  = fill_image
        self.fill_mask   = fill_mask
        self.interp_img  = interpolation_image
        self.interp_mask = interpolation_mask

    def __call__(self, image, mask):
        img_size    = TF.get_image_size(image)   # (W, H)
        size        = [img_size[1], img_size[0]]  # (H, W)
        displacement = T.ElasticTransform.get_params(self.alpha, self.sigma, size)
        image = TF.elastic_transform(image, displacement, self.interp_img,  self.fill_image)
        mask  = TF.elastic_transform(mask,  displacement, self.interp_mask, self.fill_mask)
        return image, mask


# ---------------------------------------------------------------------------
# Nur-Bild-Transforms  (Maske bleibt unverändert)
# ---------------------------------------------------------------------------

class JointColorJitter(JointTransform):
    """
    ColorJitter nur auf das Bild – die Maske bleibt unverändert.
    Bei Grayscale-Bildern ("L") werden saturation und hue automatisch
    deaktiviert, da PIL diese nicht unterstützt.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness   = brightness
        self.contrast     = contrast
        self.saturation   = saturation
        self.hue          = hue
        self._jitter_rgb  = T.ColorJitter(brightness=brightness, contrast=contrast,
                                          saturation=saturation, hue=hue)
        self._jitter_gray = T.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, image, mask):
        if image.mode == "L":
            image = self._jitter_gray(image)
        else:
            image = self._jitter_rgb(image)
        return image, mask


class JointGaussianBlur(JointTransform):
    """Gaussian Blur nur auf das Bild."""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self._blur = T.GaussianBlur(kernel_size, sigma)

    def __call__(self, image, mask):
        image = self._blur(image)
        return image, mask


class JointRandomGrayscale(JointTransform):
    """Zufällig Grayscale nur auf das Bild."""

    def __init__(self, p=0.1, num_output_channels=3):
        self.p = p
        self.num_output_channels = num_output_channels

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
        return image, mask


# ---------------------------------------------------------------------------
# Tensor-Konvertierung & Normalisierung
# ---------------------------------------------------------------------------

class JointToTensor(JointTransform):
    """
    Konvertiert PIL-Bild → Float-Tensor [0,1]
    und PIL-Maske → Long-Tensor (Klassenindizes).
    """

    def __call__(self, image, mask):
        image = TF.to_tensor(image)                             # [C, H, W], float32

        mask_np = np.array(mask, dtype=np.int64)               # [H, W]
        mask    = torch.from_numpy(mask_np)                     # Long-Tensor

        return image, mask


class JointNormalize(JointTransform):
    """
    Normalisiert nur das Bild – die Maske bleibt unverändert.
    Passt mean/std automatisch an die Kanalzahl des Tensors an:
      - 1-Kanal (Grayscale "L"): mittelt mean/std auf einen Wert
      - 3-Kanal (RGB):           verwendet alle drei Werte
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = list(mean)
        self.std  = list(std)

    def __call__(self, image, mask):
        c = image.shape[0]   # Kanalzahl des Tensors nach JointToTensor
        if c == 1 and len(self.mean) != 1:
            # RGB-Statistiken → Grayscale: Durchschnitt der 3 Werte
            mean = [sum(self.mean) / len(self.mean)]
            std  = [sum(self.std)  / len(self.std)]
        elif c == len(self.mean):
            mean, std = self.mean, self.std
        else:
            raise ValueError(
                f"JointNormalize: Bild hat {c} Kanal/Kanäle, "
                f"aber mean hat {len(self.mean)} Einträge."
            )
        image = TF.normalize(image, mean, std)
        return image, mask


# ---------------------------------------------------------------------------
# Beispiel-Dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    class SegmentationDataset(Dataset):
        def __init__(self, image_paths, mask_paths, transform=None):
            self.image_paths = image_paths
            self.mask_paths  = mask_paths
            self.transform   = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert("RGB")
            mask  = Image.open(self.mask_paths[idx])              # Graustufen / P-Mode

            if self.transform:
                image, mask = self.transform(image, mask)

            return image, mask

    # Augmentation-Pipeline
    train_transform = JointCompose([
        JointResize(256),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=30),
        JointRandomResizedCrop(size=256, scale=(0.7, 1.0)),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std =[0.229, 0.224, 0.225]),
    ])

    val_transform = JointCompose([
        JointResize(256),
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                       std =[0.229, 0.224, 0.225]),
    ])

    print("Pipeline ready:")
    print(train_transform)
    
