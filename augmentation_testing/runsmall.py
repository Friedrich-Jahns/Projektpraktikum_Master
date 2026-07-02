import torch
from torch import nn, optim
from smallunet import Unet, dice_loss
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",     type=str, required=True)
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--bs",           type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--worker",           type=int, default=8)
    parser.add_argument("--resume",    type=str, default=None,       # NEU
                    help="Run-Name zum Weitermachen (res/<name>/)")

    args = parser.parse_args()
    
    if args.resume is not None:
        resume_dir = Path("res") / args.resume
        with open(resume_dir / "config.json") as f:
            old_config = json.load(f)
        for key, value in old_config.items():
            if key in ("run_name", "epochs", "resume"):
                continue
            if hasattr(args, key):
                setattr(args, key, value)
        log_path = resume_dir / "train_log.json"
        if log_path.exists():
            with open(log_path) as f:
                old_log = json.load(f)
            train_log = list(zip(old_log["train_loss"], old_log["val_loss"]))
            best_val_loss = min(old_log["val_loss"])
        else:
            train_log = []
            best_val_loss = float("inf")
        resume_weights = resume_dir / "best_model.pth"
    else:
        train_log = []
        best_val_loss = float("inf")
        resume_weights = None

    
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
    size = 512
    crop_size = 256
    train_transform = JointCompose([
        JointResize((size, size)),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=30),
        JointRandomResizedCrop(size=(crop_size, crop_size), scale=(0.7, 1.0)),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        JointGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        JointToTensor(),
        
        JointNormalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
    ]) # JointMaskedGauss(sigmaLow=0.5, sigmaUpper=1.0),

    val_transform = JointCompose([
        JointResize((size, size)),
        JointToTensor(),
        JointNormalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
    ])
    # Daten Laden, Worker, Batch etc und augmentation festlegen
    train_dataloader, val_dataloader = get_dataloaders(
        images_dir = "dat/train/img",
        masks_dir  = "dat/train/mask",
        val_split  = 0.2,
        batch_size=args.bs, image_size=size,
        train_transform=train_transform, val_transform=val_transform,
        num_workers=args.worker,
        image_mode="L", mask_mode="L",
        prefetch=4,
    )
    
    model = Unet().to(device)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)  # einmalig vor dem Training (Test) ,mode="reduce-overhead"
    
    if resume_weights is not None:
        state_dict = torch.load(resume_weights, map_location=device)
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target.load_state_dict(state_dict)
        print(f"Gewichte aus '{args.resume}' geladen.")
    else:
        train_log = []
        best_val_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    
    scaler = GradScaler(enabled=use_amp)

    optimizer.zero_grad(set_to_none=True)
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = 0.0

        for i, batch in tqdm(enumerate(train_dataloader), leave=False):

            images, masks = batch
            images, masks = images.to(device, non_blocking=True, memory_format=torch.channels_last), masks.to(device, non_blocking=True,)
            with autocast(device_type=device.type, enabled=use_amp):
                output =  model(images).squeeze(1) # torch.sigmoid(model(imgs))
                loss = criterion(output, masks.float()) + dice_loss(output, masks.float())# criterion(outputs, masks) + dice_loss(outputs, masks)
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
                    val_outputs = model(val_imgs).squeeze(1)
                    loss_v = (
                        criterion(val_outputs, val_masks.float())
                        + dice_loss(val_outputs, val_masks.float())
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
    plt.plot(range(1, len(log_arr[0]) + 1), log_arr[0], label='train_loss')
    plt.plot(range(1, len(log_arr[1]) + 1), log_arr[1], label='val_loss')
    plt.legend()
    plt.title(f'{epoch+1}/{args.epochs}')
    plt.savefig(out_dir / 'train_log.png')
    plt.clf()

if __name__ == "__main__":
    main()