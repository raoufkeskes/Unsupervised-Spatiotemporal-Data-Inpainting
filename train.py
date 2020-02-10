import argparse
import time
import torch
import torch.nn as nn
from data.utils import getDataloaders
from models import ResnetGenerator2, Discriminator, g, d
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


def epoch(generator, discriminator_s, discriminator_f, data, criterion, L1=None, optimizer=None):
    if optimizer is None:
        generator.eval()
        discriminator_s.eval()
        discriminator_f.eval()
    else:
        generator.train()
        discriminator_s.train()
        discriminator_f.train()

    avg_loss_g = AverageMeter()
    avg_loss_d = AverageMeter()
    avg_loss = [avg_loss_d, avg_loss_g]
    avg_batch_time = AverageMeter()

    tic = time.time()


    # when d_labels==0 we are optimizing the discriminators. when d_labels==1 we are optimizing the generator.
    d_labels = 0
    for i, (y, x, idx) in enumerate(data):
        torch.cuda.empty_cache()
        y = y.to(device)
        x = x.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            x_hat = generator(y)


            # get the occlusion object used to generate y.
            occ = occ_list[idx[0].item()]

            # apply occlusion on each video in x_hat. Since the occlusion object does not expect a batch, we need to
            # loop.
            # y_hat = []--------------------------------------------------------------
            # for j in range(y.size(0)):
            #     y_hat.append(occ(x_hat[j].transpose(0, 1))[None])
            # y_hat = torch.cat(y_hat).to(device)
            # y_hat.transpose_(1, 2)
            # print("y_hat shape:", y_hat.shape)


            # get the training labels used as target in our GAN loss. Since the Discriminator and the generator have
            # opposite objectives, ascending and descending respectively. The real and fake labels depends on whether we
            # are now optimizing Ds, Df or G. Therefore, the labels depend on d_labels.
            df_hat = discriminator_f(x_hat[:, :, 0]).view(-1)
            label_real = torch.full(df_hat.shape, 1 - d_labels, device=device)
            label_fake = torch.full(df_hat.shape, d_labels, device=device)

            # Note: when optimizing the generator weights. We want the discriminator to ONLY know that generated videos
            # are real. Whereas, when optimizing the discriminator weights, we want it to know that generated videos
            # are indeed  fake AND to to know that real videos are indeed real. That's why only one BCELoss is needed
            # when optimizing the generator weights.
            loss = 0
            # 1st term of the loss: BCE on the output of the frames discriminator.
            for j in range(y.size(2)):
                loss += criterion(discriminator_f(x_hat[:, :, j]).view(-1), label_fake)
                if d_labels == 0:
                    loss += criterion(discriminator_f(x[:, :, j]).view(-1), label_real)

            # 2nd term of the loss: BCE on the output of the frames discriminator given the differences in frames this
            # time.

            if type(occ).__name__ != "RemovePixels":
                x_hat_diff = x_hat[:, :, 1:] - x_hat[:, :, :-1]
                x_diff = x[:, :, 1:] - x[:, :, :-1]
                df_diff = discriminator_f(x_diff[:, :, 0]).view(-1)
                label_real = torch.full(df_diff.shape, 1 - d_labels, device=device)
                label_fake = torch.full(df_diff.shape, d_labels, device=device)
                for j in range(x_diff.size(2)):
                    loss += criterion(discriminator_f(x_hat_diff[:, :, j]).view(-1), label_fake)
                    if d_labels == 0:
                        loss += criterion(discriminator_f(x_diff[:, :, j]).view(-1), label_real)

            # reducing over the number of frames
            loss *= 1 / y.size(2)

            # 3rd term of the loss: BCE on the output of the sequence discriminator
            ds = discriminator_s(x_hat).view(-1)
            label_real = torch.full(ds.shape, 1 - d_labels, device=device)
            label_fake = torch.full(ds.shape, d_labels, device=device)

            loss += criterion(ds, label_fake)
            if d_labels == 0:
                loss += criterion(discriminator_s(x).view(-1), label_real)
            if d_labels == 1:
                loss += L1(x, x_hat)

        if optimizer is not None:
            optimizer[d_labels].zero_grad()
            loss.backward()
            optimizer[d_labels].step()

        batch_time = time.time() - tic
        tic = time.time()

        avg_loss[d_labels].update(loss.item())
        avg_batch_time.update(batch_time)
        d_labels = 1 - d_labels
        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'D Loss {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                  'G Loss {loss_g.val:.4f} ({loss_g.avg:.4f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss_d=avg_loss[0],
                loss_g=avg_loss[1]))
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg D loss {loss_d.avg:.4f}\t'
          'Avg G loss {loss_g.avg:.4f}\n'.format(
        batch_time=int(avg_batch_time.sum), loss_d=avg_loss[0], loss_g=avg_loss[1]))

    return avg_loss_g.avg, avg_loss_d.avg


if __name__ == '__main__':

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="../datasets/KTH/", type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--num_frames', default=5, type=int, metavar='N', help='number of frames (default: 35)')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='learning rate')

    args = parser.parse_args()

    # Defining the GAN: generator, discriminator for sequence and discriminator for frames
    #G = ResnetGenerator2.define_G(3, 3, 32)
    G = g.define_G(3, 3, 32)

    #Ds = Discriminator.define_D('3', 3, 32)
    Ds = Discriminator.define_D2d(3, 32, mode='3')

    #Df = Discriminator.define_D('2', 3, 32)
    Df = Discriminator.define_D3d(3, 32, mode='2')


    print(G)
    input_g = torch.rand((2, 3, 5, 64, 64)).to(device)  # input =[Batch_size , channels, frames width , height]

    output_g = G(input_g)

    print(output_g.shape)

    # Defining the optimizers: One for the generator and one for the discriminators since they have different
    # param updates
    optimG = torch.optim.Adam(G.parameters(), args.lr, betas=(0.99, 0.99))
    optimD = torch.optim.Adam(list(Ds.parameters()) + list(Df.parameters()), args.lr, betas=(0.99, 0.99))

    # Define the GAN vanilla loss binary cross entropy. And since the generator output is not a Sigmoid we use
    # BCEWithLogitsLoss
    loss_func = nn.BCEWithLogitsLoss()

    # Basic resizing, normalization and channel transpose transforms
    img_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # get train, validation and test Dataloaders objects from a given dataset, using a given occlusion function.
    train, val, test = getDataloaders(args.root, img_transforms, occlusions=[MovingBar()], nb_frames=args.num_frames,
                                      batch_size=args.batch_size, val_size=0.05, test_size=0.05)

    # sample frames for which we keep track of the qualitative results on Tensorboard. Keep one from each set:
    # validation, train and test.
    tb_occ_video_train = next(iter(train))[0]
    tb_occ_video_val = next(iter(val))[0]
    tb_occ_video_test = next(iter(test))[0]
    tb_occ_video_train = tb_occ_video_train[0:1].to(device)
    tb_occ_video_val = tb_occ_video_val[0:1].to(device)
    tb_occ_video_test = tb_occ_video_test[0:1].to(device)

    # Create directoris if they don't already exist of:
    # Checkpoint to save training state at each epoch. model_weight to save the generator weights that give the best
    # loss on the validation set. And runs, to save the loggs for the tensoboard.
    os.makedirs(os.path.join("Checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("model_weight"), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs"), flush_secs=20)

    # Check if checkpoint exist to continue training.
    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints", 'training_state.pt'), map_location=device)
        G.load_state_dict(checkpoint['G_state_dict'])
        Ds.load_state_dict(checkpoint['Ds_state_dict'])
        Df.load_state_dict(checkpoint['Df_state_dict'])
        optimG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimD.load_state_dict(checkpoint['optimizerD_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    # start/continue training for a given number of epochs
    for e in range(starting_epoch, args.epochs):

        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")

        # return the averaged losses after doing one pass through the whole training data.
        loss_generator, loss_discriminator = epoch(G, Ds, Df, train, loss_func, L1=nn.L1Loss(), optimizer=(optimD, optimG))

        # Save a checkpoint.
        torch.save({
            'epoch': e,
            'G_state_dict': G.state_dict(),
            'Ds_state_dict': Ds.state_dict(),
            'Df_state_dict': Df.state_dict(),
            'best_loss': best_loss,
            'optimizerG_state_dict': optimG.state_dict(),
            'optimizerD_state_dict': optimD.state_dict()
        }, os.path.join("Checkpoints", 'training_state.pt'))

        loss_generator_val, loss_discriminator_val = epoch(G, Ds, Df, val, loss_func, L1 =nn.L1Loss())

        # Save generator weights if we have a better validation loss.
        if loss_generator_val < best_loss:
            best_loss = loss_generator_val
            torch.save({
                'G_state_dict': G.state_dict(),
                'Ds_state_dict': Ds.state_dict(),
                'Df_state_dict': Df.state_dict(),
                'loss_generator_val': loss_generator_val, 'loss_discriminator_val': loss_discriminator_val,
                'loss_generator': loss_generator, 'loss_discriminator': loss_discriminator
            }, os.path.join("model_weight", 'best_weight.pt'))

        loss_generator_test, loss_discriminator_test = epoch(G, Ds, Df, test, loss_func, L1=nn.L1Loss())

        # Display to tensorboard the generator performance on sampled we saved.
        with torch.no_grad():
            G.eval()

            filled_video = G(tb_occ_video_train)
            filled_video.transpose_(1, 2)
            filled_video = (filled_video + 1)/2.
            tb.add_video('train', filled_video, e)

            filled_video = G(tb_occ_video_val)
            filled_video.transpose_(1, 2)
            filled_video = (filled_video + 1)/2.
            tb.add_video('val', filled_video, e)

            filled_video = G(tb_occ_video_test)
            filled_video.transpose_(1, 2)
            filled_video = (filled_video + 1)/2.
            tb.add_video('test', filled_video, e)

        # plot the losses of train, validation and test sets.
        tb.add_scalars('Generator', {"train": loss_generator, "val": loss_generator_val, "test": loss_generator_test},
                       e)
        tb.add_scalars('Discriminator', {"train": loss_discriminator, "val": loss_discriminator_val,
                                         "test": loss_discriminator_test}, e)

    tb.close()
