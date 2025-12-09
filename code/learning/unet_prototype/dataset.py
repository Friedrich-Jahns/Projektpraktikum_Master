import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_dataloader(image_dir, mask_dir, batch_size=4, shuffle=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SegmentationDataset(image_dir, mask_dir, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
