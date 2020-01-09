"""
Created on Thu Dec 07 21:24:14 2019
@author: ilyas Aroui

A stand-alone program to calculate the the Frechet Inception Distance (FID) between two datasets distributions as
described here : https://arxiv.org/abs/1706.08500.
Usually used to evaluate GANs. Unlike the original paper, here we use Inception V3 last activations, of size 2048, as
random variables sampled from the two distribution. As indicated by the original paper, the dataset should be larger
than 2048 for correct results, 10,000 is recommended.

Some testing cases of the cov matrix squre-root are inspired from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
"""
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import numpy as np
import warnings
import scipy
from tqdm import tqdm
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def readData(batch_size, path):
    """Return a dataloader over a dataset
    :param
    batch_size (int): batch_size used during the forward pass to calculate the activations.
    path (string): full path this dataset location in the form of:
                            path_gen/images/img1.ext
                            path_gen/images/img2.ext
                            path_gen/images/img3.ext
    Where ./images is the only folder in ./data
    :returns
    A dataloder object
    """
    dataset = datasets.ImageFolder(path,
                                   transform=transforms.Compose([
                                       transforms.Resize((299, 299)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                   ]))

    loader = DataLoader(dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(),
                        num_workers=4)

    return loader


class Identity(nn.Module):
    """A layer with no parameters, or operations. Returns the input"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def getStatistics(model, loader):
    """Calculate the mean and covariance matrix over 'model' activations generated from 'loader
    :param
    model (nn.Module): inception V3 model pretrained on ImageNet with the last FC removed.
    loader (Dataloader): dataloder over the desired dataset generated from readData() function.
    :returns
    mu (float): the mean of the activation. numpy array of shape (2048,)
    sigma (float): the covariance matrix over the activations. numpy array of shape (2048, 2048)
    """
    activations = []
    for i, (images, _) in enumerate(tqdm(loader)):
        images.to(device)
        with torch.no_grad():
            activations.append(model(images).cpu().numpy())

    activations = np.concatenate(activations, axis=0)
    m = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    del activations
    return m, sigma


def inceptionModel():
    """
    A helper to prepare the inception v3 model for the activations extraction step.
    :return:
    inception (nn.Module): inception V3 model pretrained on ImageNet with the last FC removed.
    """
    inception = models.inception_v3(pretrained=True)
    inception.fc = Identity()
    for param in inception.parameters():
        param.requires_grad = False
    inception.to(device)
    inception.eval()
    return inception


def getFID(path_real, path_gen, batch_size):
    """calculate the the Frechet Inception Distance (FID) given the two paths to the dataset. Or, path_real can be
    the pre calculated mean and sigma of the real dataset.
    :param
    path_gen (string): full path the generated dataset  in the form of:
                            path_gen/images/img1.ext
                            path_gen/images/img2.ext
                            path_gen/images/img3.ext
    Where ./images is the only folder in path_gen
    path_real (string): full path to the real dataset in the a form same to path_gen. If there's another file with
    extension .npz in the folder path_real, read and use the pre-calculated mu_real and sigma_real.
    :returns
    fid (float): the Frechet Inception Distance = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2*sqrt(sigma_1*sigma_2))
    """
    eps = 1.e-6
    loader_gen = readData(batch_size, path_gen)
    inception = inceptionModel()
    m_gen, sigma_gen = getStatistics(inception, loader_gen)
    files = [f for f in os.listdir(path_real) if ".npz" in f]
    if len(files) == 0:
        loader_real = readData(batch_size, path_real)
        m_real, sigma_real = getStatistics(inception, loader_real)
        np.savez(os.path.join(path_real, "dataStat.npz"), mu=m_real, sigma=sigma_real)
    else:
        f = np.load(os.path.join(path_real, files[-1]))
        m_real, sigma_real = f['mu'], f['sigma']

    assert m_gen.shape == m_real.shape, "the means have different shapes!"
    assert sigma_gen.shape == sigma_real.shape, "the sigmas have different shapes!"

    diff = m_gen - m_real

    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_gen + offset).dot(sigma_real + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_real) - 2 * tr_covmean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_r', '-pr', type=str, metavar='DIR', help='path to real dataset')
    parser.add_argument('--path_g', '-pg', type=str, metavar='DIR', help='path to generated dataset')
    parser.add_argument('--batch_size', '-b', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    args = parser.parse_args()
    fid = getFID(args.path_r, args.path_g, args.batch_size)
    print("The Frechet Inception Distance is  %.2f." % fid)
    