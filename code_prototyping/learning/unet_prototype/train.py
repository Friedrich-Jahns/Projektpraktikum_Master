import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import get_dataloader
from unet import UNet
from pathlib import Path
from tqdm import tqdm

# Pfade zu deinen Daten
dat_path = Path.cwd().parent.parent.parent / "data" / "training" /'train'


batch_size = 4
num_epochs = 1
use_subset = True
subset_size = 20 


image_dir = dat_path / "img"
mask_dir = dat_path / "mask"

image_dir = dat_path / "fill1"
mask_dir = dat_path / "fill2"

# DataLoader
print('Load Data',end='\r')
dataloader = get_dataloader(image_dir, mask_dir, batch_size=batch_size)

# Gerät
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modell, Loss, Optimizer
print('Create model',end='\r')
model = UNet().to(device)
criterion = nn.BCELoss()  # binäre Segmentierung
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
print('Create model',end='\r')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"), 1):
        images = images.to(device)
        masks = masks.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0 or i == len(dataloader):
            avg_loss = running_loss / i
            tqdm.write(f"Batch {i}/{len(dataloader)}, avg loss: {avg_loss:.4f}")

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} finished, avg epoch loss: {epoch_loss:.4f}")


# Modell speichern

save_dir = Path("models")
save_dir.mkdir(exist_ok=True)

existing_models = list(save_dir.glob("unet_model_*.pth"))

if existing_models:
    # letzte Nummer herausfinden
    nums = [int(f.stem.split("_")[-1]) for f in existing_models]
    next_num = max(nums) + 1
else:
    next_num = 1

save_path = save_dir / f"unet_model_{next_num}.pth"

# Modell speichern
torch.save(model.state_dict(), save_path)
print(f"Modell gespeichert unter: {save_path}")
