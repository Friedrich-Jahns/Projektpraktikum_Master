import torch
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from unet import Unet

cwd = Path(__file__).resolve().parents[3]
img_path = cwd / 'data/training/train/img'
mask_path = cwd / 'data/training/train/mask'
print(cwd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#image_dir = dat_path / "img"
#mask_dir = dat_path / "mask"
model = Unet()
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model = model.to(device)
model.eval()

num_samples = 5
img_files = list(img_path.glob("*.png"))
sample_files = random.sample(img_files, num_samples)

fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
axes = np.atleast_2d(axes)

for i, img_file in enumerate(sample_files):
    img = Image.open(img_file).convert("L")
    img_np = np.array(img) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)

    mask_file = mask_path / img_file.name
    mask = Image.open(mask_file).convert("L")
    mask_np = np.array(mask)

    overlay = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = pred_bin
    overlay[..., 1] = mask_np / 255.0

    ax_orig, ax_pred, ax_mask, ax_overlay = axes[i]
    ax_orig.imshow(img_np, cmap='gray')
    ax_orig.set_title("Original")
    ax_orig.axis('off')

    ax_pred.imshow(pred_bin, cmap='Reds')
    ax_pred.set_title("Prediction")
    ax_pred.axis('off')

    ax_mask.imshow(mask_np, cmap='Greens')
    ax_mask.set_title("Mask")
    ax_mask.axis('off')

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay")
    ax_overlay.axis('off')

plt.tight_layout()
plt.savefig("predictions_overlay.png")
print("Predictions + Overlay gespeichert als predictions_overlay.png")

