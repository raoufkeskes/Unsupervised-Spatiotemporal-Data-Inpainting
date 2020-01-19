# Created by raouf at 19/01/2020


import torch
from torchvision import transforms
from data.utils import getDataloaders , write_video
from data.occlusions import RainDrops,RemovePixels,MovingBar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

positions  = [[0.1,0.1],[0.1,0.3],[0.1,0.5],[0.1,0.7],[0.1,0.9]]
height     = [0.2,0.2,0.2,0.2,0.2]
width      = 1.0/64
speed      = [0.1,0.2,0.3,0.4,0.5]

transform   = transforms.Compose([ transforms.ToTensor() ])

raindrops   = RainDrops(5,positions=positions,width=width,height=height,speed=speed)
raindrops_default  = RainDrops()

random_pix  = RemovePixels(threshold=0.05)

moving_bar  = MovingBar(position=[0.2,0.2],width=0.2,height=0.4,speed=0.05)

train_loader, val_loader, test_loader = getDataloaders ("../../datasets/FaceForensics/",transform=transform,occlusions=[moving_bar] )


for x_train_batch, y_train_batch, occ_ix in  train_loader :

    # to device
    x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)

    break

# untransform your video before saving it to get
untransformed_video = y_train_batch[2].cpu() *255

write_video( untransformed_video , "../outputs/FaceForensics/tmp","temp.mp4" )



