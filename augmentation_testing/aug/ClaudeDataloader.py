"""
segmentation_dataloader.py
--------------------------
DataLoader für Bild-Masken-Paare zum Training eines U-Nets.
Unterstützt zwei gängige Ordnerstrukturen:

  Struktur A (getrennte Ordner):          Struktur B (gemeinsamer Root):
  ├── images/                             ├── train/
  │   ├── train/                          │   ├── img001.png
  │   └── val/                            │   └── img001_mask.png
  └── masks/                              └── val/
      ├── train/
      └── val/

Verwendung:
    from segmentation_dataloader import get_dataloaders

    train_loader, val_loader = get_dataloaders(
        images_dir = "data/images/train",
        masks_dir  = "data/masks/train",
        val_images_dir = "data/images/val",
        val_masks_dir  = "data/masks/val",
        batch_size = 8,
        image_size = 256,
        num_classes = 2,
    )

    for images, masks in train_loader:
        # images: [B, 3, H, W]  float32, normalisiert
        # masks:  [B, H, W]     int64, Klassenindizes
        ...
"""

import os
import re
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from aug.ClaudeAug import (
    JointCompose,
    JointResize,
    JointRandomHorizontalFlip,
    JointRandomVerticalFlip,
    JointRandomRotation,
    JointRandomResizedCrop,
    JointColorJitter,
    JointGaussianBlur,
    JointToTensor,
    JointNormalize,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _find_mask_for(image_path: Path, masks_dir: Path,
                   mask_suffix: str = "_mask") -> Path:
    """
    Sucht die Maske zu einem Bild.
    Strategie: gleicher Dateiname (beliebige Endung) im masks_dir,
    oder Dateiname + mask_suffix.
    """
    stem = image_path.stem
    # 1) exakter Name
    for ext in IMG_EXTENSIONS:
        candidate = masks_dir / (stem + ext)
        if candidate.exists():
            return candidate
    # 2) Name + Suffix
    for ext in IMG_EXTENSIONS:
        candidate = masks_dir / (stem + mask_suffix + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Keine Maske für '{image_path.name}' in '{masks_dir}' gefunden.\n"
        f"Erwartet: '{stem}<ext>' oder '{stem}{mask_suffix}<ext>'"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Lädt (Bild, Maske)-Paare aus zwei parallelen Ordnern.

    Args:
        images_dir:   Ordner mit den Eingangsbildern.
        masks_dir:    Ordner mit den Masken (gleiche Dateinamen wie Bilder).
        transform:    JointCompose-Pipeline (oder None).
        mask_suffix:  Optionaler Suffix im Maskennamen, z.B. "_mask".
        image_mode:   PIL-Modus für Bilder ("RGB", "L", …).
        mask_mode:    PIL-Modus für Masken ("L" = Graustufen-Klassenindizes,
                      "P" = Palette).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        mask_suffix: str = "",
        image_mode: str = "RGB",
        mask_mode: str = "L",
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform
        self.mask_suffix = mask_suffix
        self.image_mode  = image_mode
        self.mask_mode   = mask_mode

        # Alle Bilder sammeln und sortieren (reproduzierbar)
        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir() if _is_image(p)
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Keine Bilder in '{images_dir}' gefunden.")

        # Maske für jedes Bild vorab suchen (gibt früh Fehler)
        self.mask_paths: List[Path] = [
            _find_mask_for(p, self.masks_dir, self.mask_suffix)
            for p in self.image_paths
        ]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert(self.image_mode)
        mask  = Image.open(self.mask_paths[idx]).convert(self.mask_mode)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SegmentationDataset(\n"
            f"  images : {self.images_dir}  ({len(self)} Samples)\n"
            f"  masks  : {self.masks_dir}\n"
            f"  transform: {self.transform}\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Hilfsmethode: Klassenverteilung analysieren
    # ------------------------------------------------------------------

    def class_counts(self, num_classes: int) -> torch.Tensor:
        """
        Zählt Pixel je Klasse über alle Masken.
        Nützlich für gewichtete CrossEntropyLoss.
        """
        counts = torch.zeros(num_classes, dtype=torch.long)
        for mask_path in self.mask_paths:
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
            for c in range(num_classes):
                counts[c] += (mask_np == c).sum()
        return counts

    def class_weights(self, num_classes: int) -> torch.Tensor:
        """
        Gibt inverse Klassenfrequenz als Gewichte zurück →
        direkt verwendbar in nn.CrossEntropyLoss(weight=...).
        """
        counts = self.class_counts(num_classes).float()
        counts = counts.clamp(min=1)           # Division durch 0 vermeiden
        weights = 1.0 / counts
        weights = weights / weights.sum()      # normalisieren
        return weights

 
# ---------------------------------------------------------------------------
# Standard-Pipelines
# ---------------------------------------------------------------------------

def build_train_transform(
    image_size: int = 256,
    mean: Optional[List[float]] = None,
    std:  Optional[List[float]] = None,
    image_mode: str = "RGB",
) -> JointCompose:
    if mean is None:
        mean = [0.5] if image_mode == "L" else [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.5] if image_mode == "L" else [0.229, 0.224, 0.225]
    return JointCompose([
        JointResize(image_size),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=30),
        JointRandomResizedCrop(size=image_size, scale=(0.75, 1.0)),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        JointToTensor(),
        JointNormalize(mean=mean, std=std),
    ])


def build_val_transform(
    image_size: int = 256,
    mean: Optional[List[float]] = None,
    std:  Optional[List[float]] = None,
    image_mode: str = "RGB",
) -> JointCompose:
    if mean is None:
        mean = [0.5] if image_mode == "L" else [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.5] if image_mode == "L" else [0.229, 0.224, 0.225]
    return JointCompose([
        JointResize(image_size),
        JointToTensor(),
        JointNormalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------

def get_dataloaders(
    # --- Pfade ---
    images_dir:     str,
    masks_dir:      str,
    val_images_dir: Optional[str] = None,
    val_masks_dir:  Optional[str] = None,
    val_split:      float = 0.15,       # nur wenn kein val_dir angegeben
    # --- Augmentation ---
    image_size:     int   = 256,
    mean: Tuple     = (0.485, 0.456, 0.406),
    std:  Tuple     = (0.229, 0.224, 0.225),
    train_transform: Optional[Callable] = None,
    val_transform:   Optional[Callable] = None,
    # --- DataLoader ---
    batch_size:     int   = 8,
    num_workers:    int   = 4,
    pin_memory:     bool  = True,
    mask_suffix:    str   = "",
    # --- Dataset-Details ---
    image_mode:     str   = "RGB",
    mask_mode:      str   = "L",
    seed:           int   = 42,
    prefetch = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Erstellt Train- und Val-DataLoader für Segmentierung.
    
    Returns:
        (train_loader, val_loader)
    """

    # Transforms
    t_train = train_transform or build_train_transform(image_size, mean, std, image_mode)
    t_val   = val_transform   or build_val_transform(image_size, mean, std, image_mode)

    # ---- Fall 1: Separate Val-Ordner ----
    if val_images_dir is not None and val_masks_dir is not None:
        train_ds = SegmentationDataset(
            images_dir, masks_dir,
            transform=t_train,
            mask_suffix=mask_suffix,
            image_mode=image_mode,
            mask_mode=mask_mode,
        )
        val_ds = SegmentationDataset(
            val_images_dir, val_masks_dir,
            transform=t_val,
            mask_suffix=mask_suffix,
            image_mode=image_mode,
            mask_mode=mask_mode,
        )

    # ---- Fall 2: Automatischer Split ----
    else:
        full_ds = SegmentationDataset(
            images_dir, masks_dir,
            transform=None,          # Transform erst nach dem Split setzen
            mask_suffix=mask_suffix,
            image_mode=image_mode,
            mask_mode=mask_mode,
        )
        n_val   = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                        generator=generator)

        # Wrapper: Transform nachträglich pro Split setzen
        train_ds = _TransformWrapper(train_ds, t_train)
        val_ds   = _TransformWrapper(val_ds,   t_val)

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     drop_last=True,           # stabilere BatchNorm
    # )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     drop_last=False,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        persistent_workers=True, 
        pin_memory=pin_memory,
        drop_last=True,       # stabilere BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        persistent_workers=True, 
        pin_memory=pin_memory,
        drop_last=False,
    )

    _print_summary(train_ds, val_ds, train_loader, val_loader, batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Interner Wrapper für nachträgliche Transforms bei random_split
# ---------------------------------------------------------------------------

class _TransformWrapper(Dataset):
    """Legt eine Transform über einen Subset, ohne das Original zu verändern."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset[idx]
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


# ---------------------------------------------------------------------------
# Konsolenausgabe
# ---------------------------------------------------------------------------

def _print_summary(train_ds, val_ds, train_loader, val_loader, batch_size):
    sep = "─" * 48
    print(sep)
    print("  Segmentation DataLoader  ")
    print(sep)
    print(f"  Train-Samples : {len(train_ds):>6}")
    print(f"  Val-Samples   : {len(val_ds):>6}")
    print(f"  Batch-Size    : {batch_size:>6}")
    print(f"  Train-Batches : {len(train_loader):>6}")
    print(f"  Val-Batches   : {len(val_loader):>6}")
    print(sep)


# ---------------------------------------------------------------------------
# Schnelltest / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil
    from PIL import ImageDraw

    # --- Synthetische Testdaten erzeugen ---
    tmp = Path(tempfile.mkdtemp())
    (tmp / "images").mkdir()
    (tmp / "masks").mkdir()

    for i in range(20):
        # Bild: buntes RGB
        img = Image.fromarray(
            np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        )
        img.save(tmp / "images" / f"sample_{i:04d}.png")

        # Maske: 2 Klassen (0 / 1)
        mask_arr = np.zeros((320, 320), dtype=np.uint8)
        mask_arr[80:240, 80:240] = 1
        Image.fromarray(mask_arr).save(tmp / "masks" / f"sample_{i:04d}.png")

    print("Synthetische Daten erstellt:", tmp)

    # --- DataLoader bauen ---
    train_loader, val_loader = get_dataloaders(
        images_dir = str(tmp / "images"),
        masks_dir  = str(tmp / "masks"),
        image_size = 256,
        batch_size = 4,
        num_workers = 0,         # 0 für einfacheres Debugging
        val_split  = 0.2,
        pin_memory = False,
    )

    # --- Einen Batch prüfen ---
    images, masks = next(iter(train_loader))
    print(f"\nBatch-Check:")
    print(f"  images : {images.shape}  dtype={images.dtype}  "
          f"min={images.min():.2f}  max={images.max():.2f}")
    print(f"  masks  : {masks.shape}  dtype={masks.dtype}  "
          f"unique={masks.unique().tolist()}")

    shutil.rmtree(tmp)
    print("\nTest erfolgreich abgeschlossen.")
