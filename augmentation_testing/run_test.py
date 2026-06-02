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

# bs 16, worker 8 -> 8,61 s per epoch
# bs 16, worker 1 -> 8,59 s per epoch ???
# mit model = model.to(memory_format=torch.channels_last)
    # model = torch.compile(model)  # einmalig vor dem Training (Test) -> 8,35 s per epoch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentation", type=str, default="baseline")
    parser.add_argument("--run_name",     type=str, required=True)
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--bs",           type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--worker",           type=int, default=1)
    args = parser.parse_args()
    # batches vorab laden
    prefetch_factor = 32   # Batches vorab laden
    
    cwd = Path(os.getcwd()).parent.parent
    img_path      = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/img'
    mask_path     = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/mask'
    img_val_path  = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/img'
    mask_val_path = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/mask'

    out_dir = Path("res") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: # Windows alternative zu cuda, schon deutlich schneller, ca faktor 5 schneller
        # natürlich immernoch deutlich langsamer als cuda wenn es funktionieren würde D:
        import torch_directml
        device = torch_directml.device()
    
    use_amp = device.type == "cuda"
    aug = load_augmentation(args.augmentation, ref_dir=img_val_path)

    train_dataloader = dataloader(img_path, mask_path, transform=aug, bs=args.bs, shuffle=True, max_samples=50, num_workers=args.worker, prefetch = prefetch_factor)
    val_dataloader   = dataloader(img_val_path, mask_val_path, bs=args.bs, shuffle=False, max_samples=30)

    model     = Unet().to(device)
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
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = 0.0

        for i, batch in tqdm(enumerate(train_dataloader), leave=False):

            images, masks = batch
            images, masks = images.to(device, non_blocking=True, memory_format=torch.channels_last), masks.to(device, non_blocking=True,)
            with autocast(device_type=device.type, enabled=use_amp):
                output =  model(images) # torch.sigmoid(model(imgs))
                loss = criterion(output, masks) + dice_loss(output, masks)# criterion(outputs, masks) + dice_loss(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs, val_masks in tqdm(val_dataloader, leave=False):
                val_imgs, val_masks = val_imgs.to(device, non_blocking=True, memory_format=torch.channels_last), val_masks.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    val_outputs = model(val_imgs)
                    loss_v = (
                        criterion(val_outputs, val_masks)
                        + dice_loss(val_outputs, val_masks)
                    )
                val_loss += loss_v.item()
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
    plt.title(f'{epoch+1}/{args.epochs} | aug: {args.augmentation}')
    plt.savefig(out_dir / 'train_log.png')
    plt.clf()

if __name__ == "__main__":
    main()