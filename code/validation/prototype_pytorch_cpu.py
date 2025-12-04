import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path

###########
#Load Model
###########
model_path = Path.cwd().parent.parent / "data" / "models"
model = torch.load(model_path/'prototype_pytorch_cpu.pth', map_location="cpu")
model.eval()



dat_path = Path.cwd().parent.parent / "data" / "training"
path = {
    "train": dat_path / "train",
    "val": dat_path / "val",
    "test": dat_path / "test",
}



def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    return img, transform(img).unsqueeze(0)   # (1, C, H, W)


# ======================================
# 5) Vorhersage durchführen
# ======================================
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # kann OrderedDict sein
        # falls output ein dict ist, nimm 'out'
        if isinstance(output, dict):
            output = output['out']
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return mask


# ======================================
# 6) Maske anzeigen
# ======================================
def show_segmentation(original, mask):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmentierung")
    plt.imshow(mask)
    plt.axis("off")

    plt.show()


# ======================================
# 7) Alles zusammen
# ======================================
if __name__ == "__main__":
    img, tensor = preprocess("/home/friedrichjahns/Msc/PP/Projektpraktikum_Master/data/training/train/img/1.png")   # <- dein Testbild
    mask = predict(tensor)
    show_segmentation(img, mask)

###########
# Validation
###########

# valid_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda img: filter_1(img)),
#     transforms.Resize((224, 224)),
# ])


# valid_dataset = CreateDataset(path['val']/'img',path['val']/'mask')
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # model.to(device)
# model.eval()

# all_labels = []
# all_preds = []

# with torch.no_grad():
#     for inputs, labels in valid_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         outputs = model(inputs)

#         _, predicted = torch.max(outputs, 1)

#         all_labels.extend(labels.cpu().numpy())
#         all_preds.extend(predicted.cpu().numpy())

# f1 = f1_score(all_labels, all_preds, average='weighted')  # Für unbalancierte Klassen ist 'weighted' sinnvoll
# print(f'F1-Score: {f1:.4f}')
