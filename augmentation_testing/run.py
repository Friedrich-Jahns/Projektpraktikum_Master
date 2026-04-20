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

parser = argparse.ArgumentParser()
parser.add_argument("--augmentation", type=str, default="baseline")
parser.add_argument("--run_name",     type=str, required=True)
parser.add_argument("--epochs",       type=int, default=100)
parser.add_argument("--bs",           type=int, default=8)
parser.add_argument("--lr",           type=float, default=1e-3)
args = parser.parse_args()

cwd = Path(os.getcwd()).parent.parent
img_path      = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/img'
mask_path     = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/train/mask'
img_val_path  = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/img'
mask_val_path = cwd / 'Projektpraktikum_Master/augmentation_testing/dat/val/mask'

out_dir = Path("res") / args.run_name
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aug = load_augmentation(args.augmentation)

train_dataloader = dataloader(img_path, mask_path, transform=aug, bs=args.bs, shuffle=True, max_samples=50)
val_dataloader   = dataloader(img_val_path, mask_val_path, bs=args.bs, shuffle=False, max_samples=30)

model     = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.BCEWithLogitsLoss()

with open(out_dir / "config.json", "w") as f:
    json.dump(vars(args), f, indent=2)

train_log = []
best_val_loss = float('inf')

for epoch in tqdm(range(args.epochs)):
    model.train()
    epoch_loss = 0.0
    for imgs, masks in tqdm(train_dataloader, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_imgs, val_masks in tqdm(val_dataloader, leave=False):
            val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)
            val_outputs = model(val_imgs)
            loss_v = criterion(val_outputs, val_masks) + dice_loss(val_outputs, val_masks)
            val_loss += loss_v.item()
    val_loss /= len(val_dataloader)

    train_log.append([epoch_loss, val_loss])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), out_dir / "best_model.pth")

    log_arr = np.array(train_log).T
    plt.plot(log_arr[0], label='train_loss')
    plt.plot(log_arr[1], label='val_loss')
    plt.legend()
    plt.title(f'{epoch+1}/{args.epochs} | aug: {args.augmentation}')
    plt.savefig(out_dir / 'train_log.png')
    plt.clf()

    print(f'{epoch+1}/{args.epochs} | train: {epoch_loss:.4f} | val: {val_loss:.4f}')

torch.save(model.state_dict(), out_dir / "last_model.pth")
# np.save(out_dir / "train_log.npy", np.array(train_log))
log_arr = np.array(train_log).T
with open(out_dir / "train_log.json", "w") as f:
    json.dump({
        "train_loss": log_arr[0].tolist(),
        "val_loss":   log_arr[1].tolist()
    }, f, indent=2)