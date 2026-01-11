import torch
import torch.nn as nn
from dataset import get_dataloader
from unet import UNet
from pathlib import Path
from tqdm import tqdm

# Pfade zu deinen Daten
dat_path = Path.cwd().parent.parent.parent / "data" / "training" /'train'


image_dir = dat_path / "img"
mask_dir = dat_path / "mask"

# DataLoader
print('Load Data',end='\r')
dataloader = get_dataloader(image_dir, mask_dir, batch_size=4)

# Gerät
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modell, Loss, Optimizer
print('Create model',end='\r')
model = UNet().to(device)
criterion = nn.BCELoss()  # binäre Segmentierung
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
print('Create model',end='\r')
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Modell speichern
torch.save(model.state_dict(), "unet_model.pth")
