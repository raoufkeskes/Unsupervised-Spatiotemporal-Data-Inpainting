import argparse
import time
import torch
from data.utils import getDataloaders
from models import ResnetGenerator, Discriminator
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from data.occlusions import RainDrops, RemovePixels, MovingBar
import os
np.random.seed(seed=1)

PRINT_INTERVAL = 20

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
occ_list = [RainDrops(), RemovePixels(), MovingBar()]

class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def epoch(generator, discriminator_s, discriminator_f, data, criterion, optimizer=None):
    if optimizer is None:
        generator.eval()
        discriminator_s.eval()
        discriminator_f.eval()
    else:
        generator.eval()
        discriminator_s.eval()
        discriminator_f.eval()

    avg_loss_g = AverageMeter()
    avg_loss_d = AverageMeter()
    avg_batch_time = AverageMeter()

    tic = time.time()
    for i, (videos, _, idx) in enumerate(data):
        videos = videos.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(imgs)
            loss = criterion(labels, pred)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc1, acc2 =accuracy(pred, labels, (1, 2))
        avg_acc1.update(acc1.item())
        avg_acc2.update(acc2.item())
        batch_time = time.time() - tic
        tic = time.time()

        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc1 {acc1.val:5.1f} ({acc1.avg:5.1f})\t'
                  'acc2 {acc2.val:5.1f} ({acc2.avg:5.1f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                acc1=avg_acc1, acc2=avg_acc2))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg_acc1 {acc1.avg:5.2f} \t'
          'Avg_acc2 {acc2.avg:5.2f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        acc1=avg_acc1, acc2=avg_acc2))

    return avg_acc1.avg, avg_acc2.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--num_frames', default=35, type=int, metavar='N', help='number of frames (default: 35)')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='learning rate')
    #parser.add_argument("--augment", help="perform data augmentation", action="store_true")

    args = parser.parse_args()
    G = ResnetGenerator.define_G(3, 3, 64)
    Ds = Discriminator.define_D('3', 3, 64)
    Df = Discriminator.define_D('2', 3, 64)

    loss_fnc = myloss

    optimG = torch.optim.Adam(G.parameters(), args.lr, betas=(0.99, 0.99))
    optimD = torch.optim.Adam(list(Ds.parameters()) + list(Df.parameters()), args.lr,betas=(0.99, 0.99))

    img_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train, val, test = getDataloaders(args.root, img_transforms, occlusions=[RainDrops()], nb_frames=args.num_frames,
                                      batch_size=args.batch_size, val_size=0.05, test_size=0.05)

    tb_occ_video_train = next(iter(train))[0]
    tb_occ_video_val = next(iter(val))[0]
    tb_occ_video_test = next(iter(test))[0]
    tb_occ_video_train = tb_occ_video_train[0:1].to(device)
    tb_occ_video_val = tb_occ_video_val[0:1].to(device)
    tb_occ_video_test = tb_occ_video_test[0:1].to(device)

    os.makedirs(os.path.join("Checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("model_weight"), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs"), flush_secs=20)

    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints", 'training_state.pt'),  map_location=device)
        G.load_state_dict(checkpoint['G_state_dict'])
        Ds.load_state_dict(checkpoint['Ds_state_dict'])
        Df.load_state_dict(checkpoint['Df_state_dict'])
        optimG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimD.load_state_dict(checkpoint['optimizerD_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    for e in range(starting_epoch, args.epochs):

        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")

        loss_generator, loss_discriminator = epoch(G, Ds, Df, train, loss_fnc, (optimG, optimD))

        torch.save({
            'epoch': e,
            'G_state_dict': G.state_dict(),
            'Ds_state_dict': Ds.state_dict(),
            'Df_state_dict': Df.state_dict(),
            'best_loss': best_loss,
            'optimizerG_state_dict': optimG.state_dict(),
            'optimizerD_state_dict': optimD.state_dict()
        }, os.path.join("Checkpoints", 'training_state.pt'))

        loss_generator_val, loss_discriminator_val = epoch(G, Ds, Df, val, loss_fnc)

        if loss_generator_val < best_loss:
            best_loss = loss_generator_val
            torch.save({
                'G_state_dict': G.state_dict(),
                'Ds_state_dict': Ds.state_dict(),
                'Df_state_dict': Df.state_dict(),
                'loss_generator_val': loss_generator_val, 'loss_discriminator_val': loss_discriminator_val,
                'loss_generator': loss_generator, 'loss_discriminator': loss_discriminator
            }, os.path.join("model_weight", 'best_weight.pt'))

        loss_generator_test, loss_discriminator_test = epoch(G, Ds, Df, test, loss_fnc)
        with torch.no_grad():
            G.eval()

            filled_video = G(tb_occ_video_train)
            tb.add_video('train', filled_video, e)

            filled_video = G(tb_occ_video_val)
            tb.add_video('val', filled_video, e)

            filled_video = G(tb_occ_video_test)
            tb.add_video('test', filled_video, e)

        tb.add_scalars('Generator', {"train": loss_generator, "val": loss_generator_val, "test": loss_generator_test},
                       e)
        tb.add_scalars('Discriminator', {"train": loss_discriminator, "val": loss_discriminator_val,
                                         "test": loss_discriminator_test}, e)

    tb.close()
