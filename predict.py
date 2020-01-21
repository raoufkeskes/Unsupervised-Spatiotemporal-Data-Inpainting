import argparse
import numpy as np
from models import ResnetGenerator
from torchvision import transforms
import os
import torch
from utils import read_video, save_video

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="inference", type=str, metavar='DIR', help='path to get images')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    file_names = sorted(os.listdir(args.path))

    mymodel = ResnetGenerator()

    os.makedirs(os.path.join("result"), exist_ok=True)
    mymodel.load_state_dict(torch.load(os.path.join("model_weight", 'best_weight.pt'),
                                       map_location=device)['G_state_dict'])

    mymodel.eval()
    imgs_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    for i in range(len(file_names)):
        video = read_video(file_names[i])
        with torch.no_grad():
            flow = mymodel(video)

        save_video(video.cpu().numpy(), os.path.join("result", file_names[i], '_filled.mp4'))
