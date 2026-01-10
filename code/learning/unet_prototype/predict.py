import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from unet import UNet
from pathlib import Path
import napari
import numpy as np
# --- Konfiguration ---


device = "cuda" if torch.cuda.is_available() else "cpu"

dat_path = Path.cwd().parent.parent.parent / "data" / "training" /'train'

model_path = "unet_model.pth"
image_path = dat_path / "img" / "PE-2025-01953-M_00_s0060_PM_Complete_Transmittance_Stitched_Flat_v000_128_3584.png"  # dein Bild

# --- Bild laden ---
transform = transforms.Compose([
    transforms.ToTensor()
])
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # Batch dimension

# --- Modell laden ---
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Vorhersage ---
with torch.no_grad():
    output = model(input_tensor)
    output_np = output.squeeze().cpu().numpy()

thresh = np.median(output_np)
mask = (output_np > thresh).astype(np.uint8)


mask_display = mask * 255

# # --- Maske anzeigen ---
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.title("Originalbild")

# plt.subplot(1,2,2)
# plt.imshow(mask, cmap="gray")
# plt.title("Segmentierungsmaske")
# plt.show()


viewer = napari.Viewer()
viewer.add_image(np.array(image), name='Originalbild')
viewer.add_image(np.array(mask_display), name='Segmentierungsmaske')
napari.run()
