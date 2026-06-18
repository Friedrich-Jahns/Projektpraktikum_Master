import torch
from torch import nn, optim
from unet import Unet, dice_loss
from dataset import dataloader
from aug import load_augmentation
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import json
import argparse
# New
from torch.amp import autocast, GradScaler
# from torch.utils.data import Dataset, DataLoader
from aug.ClaudeAug import *
from aug.ClaudeDataloader import *
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",     type=str, required=True)
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--bs",           type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-7)
    parser.add_argument("--worker",           type=int, default=4)
    args = parser.parse_args()
    
    cwd = Path(os.getcwd()).parent.parent
    # img_path      = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/img'
    # mask_path     = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/mask'
    # img_val_path  = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/img'
    # mask_val_path = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/mask'

    out_dir = Path("res") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: # Windows alternative zu cuda, schon deutlich schneller, ca faktor 5 schneller
        # natürlich immernoch deutlich langsamer als cuda wenn es funktionieren würde D:
        import torch_directml
        device = torch_directml.device()
    
    use_amp = device.type == "cuda"
    # Augmentation-Pipeline
    train_transform = JointCompose([
        JointResize((512, 512)),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=30),
        JointRandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        JointToTensor(),
        JointMaskedGauss(sigmaLow=0.5, sigmaUpper=1.0),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
    ])# JointMaskedGauss(sigmaLow=0.5, sigmaUpper=1.0)

    val_transform = JointCompose([
        JointResize((512, 512)),
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
    ])
    # Daten Laden, Worker, Batch etc und augmentation festlegen
    train_dataloader, val_dataloader = get_dataloaders(
        images_dir = "dat/train/img",
        masks_dir  = "dat/train/mask",
        val_split  = 0.15,
        batch_size=args.bs, image_size=512,
        train_transform=train_transform, val_transform=val_transform,
        num_workers=args.worker,
        image_mode="L", mask_mode="L",
        prefetch=2,
    )
    
    model = Unet().to(device)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)  # einmalig vor dem Training (Test) ,mode="reduce-overhead"
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    # torch.backends.cudnn.benchmark = True
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_log = []
    best_val_loss = float('inf')
    scaler = GradScaler(enabled=use_amp)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
    
        t_epoch_start = time.perf_counter()
        t_data_total = 0.0
        t_to_device_total = 0.0
        t_forward_backward_total = 0.0
        t_item_total = 0.0
    
        t_last = time.perf_counter()
    
        for i, batch in enumerate(train_dataloader):
            t_after_load = time.perf_counter()
            t_data_total += (t_after_load - t_last)
    
            images, masks = batch
            images, masks = (
                images.to(device, non_blocking=True, memory_format=torch.channels_last),
                masks.to(device, non_blocking=True),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_after_to_device = time.perf_counter()
            t_to_device_total += (t_after_to_device - t_after_load)
    
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(images).squeeze(1)
                loss = criterion(output, masks.float()) + dice_loss(output, masks.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_after_compute = time.perf_counter()
            t_forward_backward_total += (t_after_compute - t_after_to_device)
    
            epoch_loss += loss.item()
            t_after_item = time.perf_counter()
            t_item_total += (t_after_item - t_after_compute)
    
            t_last = time.perf_counter()
    
            # Nur die ersten 2 Epochen genau loggen, dann normal weiterlaufen
            if epoch == 0 and i == len(train_dataloader) - 1:
                t_train_total = time.perf_counter() - t_epoch_start
                print(f"\n{'='*50}")
                print(f"DIAGNOSE EPOCHE {epoch+1} (TRAIN, {len(train_dataloader)} Batches)")
                print(f"{'='*50}")
                print(f"  Gesamt              : {t_train_total:.2f}s")
                print(f"  Datenladen (wait)   : {t_data_total:.2f}s  ({t_data_total/t_train_total*100:.1f}%)")
                print(f"  .to(device)         : {t_to_device_total:.2f}s  ({t_to_device_total/t_train_total*100:.1f}%)")
                print(f"  forward+backward    : {t_forward_backward_total:.2f}s  ({t_forward_backward_total/t_train_total*100:.1f}%)")
                print(f"  loss.item() sync    : {t_item_total:.2f}s  ({t_item_total/t_train_total*100:.1f}%)")
                summe = t_data_total + t_to_device_total + t_forward_backward_total + t_item_total
                print(f"  SUMME aller Teile   : {summe:.2f}s  (sollte ~Gesamt entsprechen)")
                print(f"{'='*50}\n")
    
        epoch_loss /= len(train_dataloader)
        model.eval()
        val_loss = 0.0
    
        t_val_start = time.perf_counter()
        with torch.no_grad():
            for val_imgs, val_masks in val_dataloader:
                val_imgs, val_masks = val_imgs.to(device, non_blocking=True, memory_format=torch.channels_last), val_masks.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    val_outputs = model(val_imgs).squeeze(1)
                    loss_v = (
                        criterion(val_outputs, val_masks.float())
                        + dice_loss(val_outputs, val_masks.float())
                    )
                val_loss += loss_v.item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_val_total = time.perf_counter() - t_val_start
        val_loss /= len(val_dataloader)


        train_log.append([epoch_loss, val_loss])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pth")

        log_arr = np.array(train_log).T
        #plt.plot(log_arr[0], label='train_loss')
        #plt.plot(log_arr[1], label='val_loss')
        #plt.legend()
        #plt.title(f'{epoch+1}/{args.epochs} | aug: {args.augmentation}')
        #plt.savefig(out_dir / 'train_log.png')
        #plt.clf()

        print(f'{epoch+1}/{args.epochs} | train: {epoch_loss:.4f} | val: {val_loss:.4f}')
                    # _orig_mod ist wegen dem torch.compile(model) ein präfix der den parametern hinzugefügt wird 
                    # und beim einlesen der parameter stört
    torch.save(model._orig_mod.state_dict(), out_dir / "last_model.pth")
    # np.save(out_dir / "train_log.npy", np.array(train_log))
    log_arr = np.array(train_log).T
    with open(out_dir / "train_log.json", "w") as f:
        json.dump({
            "train_loss": log_arr[0].tolist(),
            "val_loss":   log_arr[1].tolist()
        }, f, indent=2)
    plt.plot(range(1, args.epochs + 1), log_arr[0], label='train_loss')
    plt.plot(range(1, args.epochs + 1), log_arr[1], label='val_loss')
    plt.legend()
    plt.title(f'{epoch+1}/{args.epochs}')
    plt.savefig(out_dir / 'train_log.png')
    plt.clf()

if __name__ == "__main__":
    main()