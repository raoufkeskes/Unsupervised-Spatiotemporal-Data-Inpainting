import argparse
import numpy as np
from models import ResnetGenerator
from torchvision import transforms
import os
import torch
from utils import read_video, save_video
import opencv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="inference", type=str, metavar='DIR', help='path to get images')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    file_names = sorted(os.listdir(args.path))

    mymodel = ResnetGenerator()
    mymodel.to(device)

    os.makedirs(os.path.join("result"), exist_ok=True)
    mymodel.load_state_dict(torch.load(os.path.join("model_weight", 'best_weight.pt'),
                                       map_location=device)['G_state_dict'])

    mymodel.eval()

    for i in range(len(file_names)):
        video = read_video(os.path.join(args.path, file_names[i]), inference=True).to(device)
        with torch.no_grad():
            reconstructed = []
            for j in range(video.size(0)):
                reconstructed.append(mymodel(video[j][None]).cpu().numpy())
        reconstructed = np.concatenate(reconstructed)
        save_video(reconstructed, os.path.join("result", file_names[i], '_filled.avi'))
