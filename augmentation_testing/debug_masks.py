"""
debug_masks.py
---------------
Diagnose-Skript: prüft die TATSÄCHLICHEN Werte, die aus train_dataloader
herauskommen -- direkt vor dem Loss. Bitte unverändert in dein Projekt
legen und ausführen (gleiche Ordnerstruktur wie Urun.py).
"""

import torch
from aug.ClaudeAug import *
from aug.ClaudeDataloader import *

train_transform = JointCompose([
    JointResize((512, 512)),
    JointRandomHorizontalFlip(p=0.5),
    JointRandomVerticalFlip(p=0.5),
    JointRandomRotation(degrees=30),
    JointRandomResizedCrop(size=256, scale=(0.7, 1.0)),
    JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    JointToTensor(),
    JointMaskedGauss(sigmaLow=0.5, sigmaUpper=1.0),
    JointNormalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
])

train_dataloader, val_dataloader = get_dataloaders(
    images_dir = "dat/train/img",
    masks_dir  = "dat/train/mask",
    val_split  = 0.15,
    batch_size = 4, image_size=512,
    train_transform=train_transform,
    num_workers=1,
    image_mode="L", mask_mode="L",
)

print("=" * 60)
print("PRÜFE 5 BATCHES AUS DEM TRAIN_DATALOADER")
print("=" * 60)

for i, (images, masks) in enumerate(train_dataloader):
    print(f"\n--- Batch {i} ---")
    print(f"images: shape={tuple(images.shape)}  dtype={images.dtype}  "
          f"min={images.min().item():.4f}  max={images.max().item():.4f}  "
          f"mean={images.mean().item():.4f}")
    print(f"masks : shape={tuple(masks.shape)}  dtype={masks.dtype}  "
          f"min={masks.min().item():.4f}  max={masks.max().item():.4f}  "
          f"unique={torch.unique(masks).tolist()}")

    # Kritische Prüfung: liegen ALLE Maskenwerte in {0, 1}?
    mask_f = masks.float()
    valid = torch.all((mask_f == 0) | (mask_f == 1))
    print(f"  -> Maske enthält NUR 0/1 Werte: {valid.item()}")

    if not valid.item():
        bad_vals = torch.unique(mask_f)
        print(f"  !! UNGÜLTIGE WERTE GEFUNDEN: {bad_vals.tolist()}")

    # Vordergrund-Anteil
    fg_ratio = (mask_f == 1).float().mean().item()
    print(f"  -> Vordergrund-Anteil: {fg_ratio*100:.2f}%")

    if i >= 4:
        break

print("\n" + "=" * 60)
print("FERTIG. Falls 'UNGÜLTIGE WERTE GEFUNDEN' erscheint,")
print("ist das die Ursache des negativen Loss.")
print("=" * 60)