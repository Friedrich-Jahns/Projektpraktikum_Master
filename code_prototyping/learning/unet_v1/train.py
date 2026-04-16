import torch
from torch import nn,optim
from unet import Unet,dice_loss
from dataset import dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1
learning_rate = 1e-3
epochs = 5

device = torch.device(cuda if torch.cuda.is_available() else 'cpu')

img_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training/train/img'
mask_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training/train/mask'


img_val_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training_mini/val/img'
mask_val_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training_mini/val/mask'

train_dataloader = dataloader(img_path,mask_path,bs=batch_size,shuffle=True,max_samples=20)
val_dataloader = dataloader(img_val_path,mask_val_path,bs=batch_size,shuffle=True,max_samples=4) 

model = Unet().to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)

train_log = []

for epoch in tqdm(range(epochs)):
    model.train()
    epoch_loss = 0.0
    for imgs,masks in tqdm(train_dataloader):
        imgs = imgs.to(device)
        mask = masks.to(device) 
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = dice_loss(outputs,masks)
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
            val_loss = dice_loss(val_output,val_masks)
            validation_loss += val_loss.item()
        validation_loss/=len(val_dataloader)
    train_log.append([epoch_loss,validation_loss])
    print(f'{epoch+1}/{epochs},trainloss:{epoch_loss:.4f},val_loss{validation_loss:.4f}')
torch.save(model.state_dict(), "unet_model.pth")

train_log = np.array(train_log).T
plt.plot(train_log[0],label='train_loss')
plt.plot(train_log[1],label='val_loss')
plt.legend()
plt.savefig('train_log')

