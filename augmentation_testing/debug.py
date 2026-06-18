# """
# benchmark_bottleneck.py
# ------------------------
# Misst getrennt:
#   1. Wie lange reines Daten-Laden + Augmentation pro Epoche dauert
#      (DataLoader allein, ohne Modell/GPU)
#   2. Wie lange die GPU-Berechnung pro Batch dauert
#      (Forward+Backward, mit vorab in den Speicher geladenen Batches)

# So lässt sich erkennen, ob der Flaschenhals beim CPU-seitigen
# Laden/Augmentieren liegt oder bei der eigentlichen GPU-Rechenzeit.

# Einfach mit den echten Pfaden/Pipeline in Urun.py-Stil ausführen.
# """

# import time
# import torch
# from aug.ClaudeAug import *
# from aug.ClaudeDataloader import *
# from unet import Unet, dice_loss


# def benchmark_dataloader_only(train_dataloader, n_batches=20):
#     """Misst NUR das Laden+Augmentieren, ohne Modell/GPU."""
#     print(f"\n--- Benchmark: DataLoader allein ({n_batches} Batches) ---")
#     t0 = time.perf_counter()
#     count = 0
#     for i, (images, masks) in enumerate(train_dataloader):
#         count += 1
#         if i + 1 >= n_batches:
#             break
#     t1 = time.perf_counter()
#     dt = t1 - t0
#     print(f"  {count} Batches in {dt:.2f}s  ->  {dt/count*1000:.1f} ms/Batch")
#     return dt / count


# def benchmark_gpu_only(model, criterion, device, batch_size, image_size,
#                         in_channels=1, n_batches=20, use_amp=True):
#     """
#     Misst NUR Forward+Backward auf der GPU, mit zufälligen Tensoren
#     (kein echtes Laden/Augmentieren -- isoliert die reine Rechenzeit).
#     """
#     from torch.amp import autocast, GradScaler
#     scaler = GradScaler(enabled=use_amp)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     dummy_images = torch.randn(batch_size, in_channels, image_size, image_size,
#                                 device=device).to(memory_format=torch.channels_last)
#     dummy_masks  = torch.randint(0, 2, (batch_size, image_size, image_size),
#                                  device=device).float()

#     # Warmup (erste Iterationen sind oft langsamer wegen CUDA-Init/torch.compile)
#     for _ in range(3):
#         with autocast(device_type=device.type, enabled=use_amp):
#             out = model(dummy_images).squeeze(1)
#             loss = criterion(out, dummy_masks) + dice_loss(out, dummy_masks)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)

#     torch.cuda.synchronize() if device.type == "cuda" else None

#     print(f"\n--- Benchmark: GPU allein ({n_batches} Batches, bs={batch_size}, size={image_size}) ---")
#     t0 = time.perf_counter()
#     for _ in range(n_batches):
#         with autocast(device_type=device.type, enabled=use_amp):
#             out = model(dummy_images).squeeze(1)
#             loss = criterion(out, dummy_masks) + dice_loss(out, dummy_masks)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)
#     torch.cuda.synchronize() if device.type == "cuda" else None
#     t1 = time.perf_counter()
#     dt = t1 - t0
#     print(f"  {n_batches} Batches in {dt:.2f}s  ->  {dt/n_batches*1000:.1f} ms/Batch")
#     return dt / n_batches


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     use_amp = device.type == "cuda"

#     # --- gleiche Pipeline wie in Urun.py ---
#     train_transform = JointCompose([
#         JointResize((512, 512)),
#         JointRandomHorizontalFlip(p=0.5),
#         JointRandomVerticalFlip(p=0.5),
#         JointRandomRotation(degrees=30),
#         JointRandomResizedCrop(size=256, scale=(0.7, 1.0)),
#         JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#         JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
#         JointToTensor(),
#         JointMaskedGauss(sigmaLow=0.5, sigmaUpper=1.0),
#         JointNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     BATCH_SIZE = 16
#     IMAGE_SIZE = 256

#     for nw in [1, 2, 4, 8]:
#         train_dataloader, _ = get_dataloaders(
#             images_dir="dat/train/img",
#             masks_dir="dat/train/mask",
#             val_split=0.15,
#             batch_size=BATCH_SIZE, image_size=512,
#             train_transform=train_transform,
#             num_workers=nw,
#             image_mode="L", mask_mode="L",
#         )
#         print(f"\n========== num_workers={nw} ==========")
#         ms_per_batch = benchmark_dataloader_only(train_dataloader, n_batches=20)
#         print(f"  -> Hochgerechnet auf volle Epoche (~{len(train_dataloader)} Batches): "
#               f"{ms_per_batch * len(train_dataloader):.1f}s")

#     # --- GPU allein, zum Vergleich ---
#     model = Unet().to(device).to(memory_format=torch.channels_last)
#     criterion = torch.nn.BCEWithLogitsLoss()
#     benchmark_gpu_only(model, criterion, device, BATCH_SIZE, IMAGE_SIZE,
#                        in_channels=1, n_batches=20, use_amp=use_amp)

"""
benchmark_real_epoch.py
-------------------------
Misst die ECHTE Trainingsepoche (mit deinem echten Modell, echten Daten,
echtem train+val Loop) und schlüsselt die Zeit auf in:
  - reine Datenladezeit (Wartezeit auf next(dataloader))
  - reine GPU-Rechenzeit (forward+backward+step)
  - Validierungszeit gesamt

So sehen wir exakt, wo die ~100s hingehen, statt zu schätzen.
Einfach in dein Projekt legen und mit echten Pfaden/Args ausführen,
z.B.: python benchmark_real_epoch.py --bs 16 --worker 8
"""

import time
import argparse
import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from unet import Unet, dice_loss
from aug.ClaudeAug import *
from aug.ClaudeDataloader import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

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
        JointNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = JointCompose([
        JointResize((512, 512)),
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataloader, val_dataloader = get_dataloaders(
        images_dir="dat/train/img",
        masks_dir="dat/train/mask",
        val_split=0.15,
        batch_size=args.bs, image_size=512,
        train_transform=train_transform, val_transform=val_transform,
        num_workers=args.worker,
        image_mode="L", mask_mode="L",
    )

    model = Unet().to(device).to(memory_format=torch.channels_last)
    # WICHTIG: für diesen Benchmark torch.compile() WEGLASSEN,
    # um Kompilier-Overhead nicht mit echter Rechenzeit zu vermischen.
    # Falls du wissen willst, ob torch.compile selbst der Übeltäter ist,
    # führe das Skript einmal MIT und einmal OHNE die nächste Zeile aus.
    # model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=use_amp)

    print(f"\nGerät: {device}  |  AMP: {use_amp}  |  bs={args.bs}  |  workers={args.worker}")
    print(f"Train-Batches: {len(train_dataloader)}  |  Val-Batches: {len(val_dataloader)}\n")

    # ---------------- EINE Epoche, fein vermessen ----------------
    model.train()
    t_epoch_start = time.perf_counter()

    t_data_total = 0.0
    t_compute_total = 0.0
    t_last = time.perf_counter()
    
    optimizer.zero_grad(set_to_none=True)
    for i, (images, masks) in enumerate(train_dataloader):
        t_after_load = time.perf_counter()
        t_data_total += (t_after_load - t_last)

        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks  = masks.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            output = model(images).squeeze(1)
            loss = criterion(output, masks.float()) + dice_loss(output, masks.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()   # WICHTIG für korrekte Zeitmessung
        t_after_compute = time.perf_counter()
        t_compute_total += (t_after_compute - t_after_load)

        t_last = time.perf_counter()

    t_train_total = time.perf_counter() - t_epoch_start
    print(f"--- TRAIN ---")
    print(f"  Gesamt          : {t_train_total:.2f}s")
    print(f"  davon Datenladen: {t_data_total:.2f}s  ({t_data_total/t_train_total*100:.1f}%)")
    print(f"  davon GPU-Compute: {t_compute_total:.2f}s  ({t_compute_total/t_train_total*100:.1f}%)")

    # ---------------- Validation, fein vermessen ----------------
    model.eval()
    t_val_start = time.perf_counter()
    t_val_data = 0.0
    t_val_compute = 0.0
    t_last = time.perf_counter()

    with torch.no_grad():
        for val_imgs, val_masks in val_dataloader:
            t_after_load = time.perf_counter()
            t_val_data += (t_after_load - t_last)

            val_imgs  = val_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            val_masks = val_masks.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                val_out = model(val_imgs).squeeze(1)
                loss_v = criterion(val_out, val_masks.float()) + dice_loss(val_out, val_masks.float())

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_after_compute = time.perf_counter()
            t_val_compute += (t_after_compute - t_after_load)
            t_last = time.perf_counter()

    t_val_total = time.perf_counter() - t_val_start
    print(f"\n--- VAL (image_size=512!) ---")
    print(f"  Gesamt          : {t_val_total:.2f}s")
    print(f"  davon Datenladen: {t_val_data:.2f}s")
    print(f"  davon GPU-Compute: {t_val_compute:.2f}s")

    print(f"\n--- GESAMT EPOCHE (train+val) ---")
    print(f"  {t_train_total + t_val_total:.2f}s")


if __name__ == "__main__":
    main()