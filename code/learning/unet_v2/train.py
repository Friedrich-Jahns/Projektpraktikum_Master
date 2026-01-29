import torch
from torch import nn,optim
from unet import Unet,dice_loss
from dataset import dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


batch_size = 8
learning_rate = 1e-3
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

cwd = Path(os.getcwd()).parent.parent.parent.parent

img_path = cwd / 'Projektpraktikum_Master/data/training/train/img'
mask_path = cwd / 'Projektpraktikum_Master/data/training/train/mask'


img_val_path = cwd / 'Projektpraktikum_Master/data/training/train/img'
mask_val_path = cwd  /'Projektpraktikum_Master/data/training/train/mask'

train_dataloader = dataloader(img_path,mask_path,bs=batch_size,shuffle=True,max_samples=100)
val_dataloader = dataloader(img_val_path,mask_val_path,bs=batch_size,shuffle=True,max_samples=30) 

model = Unet().to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)

train_log = []

criterion_bce = torch.nn.BCEWithLogitsLoss()

for epoch in tqdm(range(epochs)):
    model.train()
    epoch_loss = 0.0
    for imgs,masks in tqdm(train_dataloader):
        imgs = imgs.to(device)
        mask = masks.to(device) 
        
        optimizer.zero_grad()
        outputs = model(imgs)
        masks = masks.to(outputs.device)
        loss = criterion_bce(outputs, masks) + dice_loss(outputs,masks)
        loss.backward()
        optimizer.step()

        
        epoch_loss += loss.item()
    epoch_loss/=len(train_dataloader)

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():
        for val_imgs, val_masks in tqdm(val_dataloader):
            val_imgs = val_imgs.to(device)
            val_masks = val_masks.to(device)

            val_output = model(val_imgs)
            val_loss = criterion_bce(val_outputs, val_masks) + dice_loss(val_output,val_masks)
            validation_loss += val_loss.item()
        validation_loss/=len(val_dataloader)
    train_log.append([epoch_loss,validation_loss])

    train_log_plot = np.array(train_log).T
    plt.plot(train_log_plot[0],label='train_loss')
    plt.plot(train_log_plot[1],label='val_loss')
    plt.legend()
    plt.title(f'{epoch+1}/{epochs}')
    plt.savefig('train_log')
    plt.clf()

    print(f'{epoch+1}/{epochs},trainloss:{epoch_loss:.4f},val_loss{validation_loss:.4f}')
torch.save(model.state_dict(), "unet_model.pth")

