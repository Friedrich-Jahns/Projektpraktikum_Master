import torch
from torch import nn,optim
from unet import Unet,dice_loss
from dataset import dataloader
from tqdm import tqdm

batch_size = 4
learning_rate = 1e-3
epochs = 10

device = torch.device(cuda if torch.cuda.is_available() else 'cpu')

img_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training/train/img'
mask_path = '/home/friedrich/Dokumente/Master/PP/Projektpraktikum_Master/data/training/train/mask'

dataloader = dataloader(img_path,mask_path,bs=batch_size,shuffle=True)

model = Unet().to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in tqdm(range(epochs)):
    model.train()
    epoch_loss = 0.0
    loss_log = []
    for imgs,masks in tqdm(dataloader):
        imgs = imgs.to(device)
        mask = masks.to(device) 
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = dice_loss(outputs,masks)
        loss.backward()
        optimizer.step()

        
        loss_log.append(loss)
        epoch_loss += loss.item()

    print(f'{epoch+1}/{num_epochs},loss:{epoch_loss/len(dataloader):.4f}')
torch.save(model.state_dict(), "unet_model.pth")


