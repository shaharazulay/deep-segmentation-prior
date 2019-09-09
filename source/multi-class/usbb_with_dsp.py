"""

Based on the code from the git repository of the paper "Unsupervised Segmentation By Backpropagation":
https://github.com/kanezaki/pytorch-unsupervised-segmentation

"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from skimage import segmentation
import scipy.io as sio
from tqdm import tqdm

from net.skip_model import *
from utils.image_io import prepare_image, pil_to_np, get_noise
from utils.torch_utils import weights_init


parser = argparse.ArgumentParser(description='USBB with DSP')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--aeIter', metavar='AE_T', default=100, type=int,
                    help='number of iterations for reconstruction')
parser.add_argument('--aeSamples', metavar='AE_S', default=10, type=int,
                    help='number of samples from reconstruction')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--dsp_lr', metavar='DSP_LR', default=0.01, type=float,
                    help='learning rate for DSP')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--nSuperpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='INPUT_FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--load_prior', metavar='PRIOR_FILENAME',
                    help='prior image file name')

parser.add_argument('--output_dir', metavar='OUTPUT_DIR',
                    help='output dir', default="./")
parser.add_argument('--output', metavar='OUTPUT_FILENAME',
                    help='output image file name', required=True)
parser.add_argument('--ae_output', metavar='AE_OUTPUT_FILENAME',
                    help='ae output image file name', required=True)

parser.add_argument('--warmup', action='store_true')
parser.add_argument("--wuSteps", type=int, default=100)

parser.add_argument('--cosine_sched', action='store_true')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_segmentation(model, data, im_shape, label_colours, args):
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % label_colours.shape[0]] for c in im_target])
    im_target = im_target.reshape(im_shape[:2]).astype(np.uint16)
    im_target_rgb = im_target_rgb.reshape(im_shape).astype(np.uint8)

    return im_target_rgb, im_target


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


def train_iter(model, optimizer, loss_fn, label_colours):
    optimizer.zero_grad()

    # forwarding
    output = model(im_torch)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        # TODO: use create_segmentation
        im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im_shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
    target = torch.from_numpy(im_target).to(device)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    return loss, nLabels


run_dir = os.path.join(args.output_dir, "runs")
run_dir = os.path.join(run_dir, time.strftime("%Y-%m-%d__%H-%M-%S"))
os.makedirs(run_dir)
tensorboard_dir = os.path.join(run_dir, "tensorboard")
os.makedirs(tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)

# load image
im = prepare_image(args.input)
im_torch = torch.from_numpy(im).unsqueeze(0).to(device)
im = im.transpose(1, 2, 0)
im_shape = im.shape

# noise
noise = get_noise(2, 'noise', (im_shape[0], im_shape[1])).to(device)

# slic
if args.load_prior is None:
    # train prior model - average over samples
    rec_samples = []
    for ae_sample in range(1, args.aeSamples + 1):
        ae = skip(noise.size(1), im_torch.size(1),
                  num_channels_down=[8, 16, 32],
                  num_channels_up=[8, 16, 32],
                  num_channels_skip=[0, 0, 0],
                  upsample_mode='bilinear',
                  filter_size_down=3,
                  filter_size_up=3,
                  need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

        optimizer = optim.Adam(ae.parameters(), lr=args.dsp_lr)
        loss_fn = torch.nn.L1Loss()

        print("--- Train AE sample % d-----" % ae_sample)
        for _ in tqdm(range(args.aeIter)):
            optimizer.zero_grad()

            loss = loss_fn(ae(noise), im_torch)

            loss.backward()

            optimizer.step()

        rec_samples.append(ae(noise).detach())

    rec_im = torch.median(torch.cat(rec_samples), dim=0)[0].permute(1, 2, 0).cpu().numpy()
    rec_im = (rec_im * 255).astype(np.uint8)
    rec_im_bgr = cv2.cvtColor(rec_im, cv2.COLOR_RGB2BGR)
    if args.visualize:
        cv2.imshow("rec_img", rec_im_bgr)
        cv2.waitKey()
    cv2.imwrite(os.path.join(run_dir, args.ae_output), rec_im_bgr)
else:
    rec_im = cv2.imread(args.load_prior)
    rec_im = cv2.cvtColor(rec_im, cv2.COLOR_BGR2RGB)

# interpolate rec_im to original image size
im = cv2.imread(args.input)
im_shape = im.shape
rec_im = cv2.resize(rec_im, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_AREA)

labels = segmentation.slic(rec_im, compactness=args.compactness, n_segments=args.nSuperpixels)
labels = labels.reshape(im_shape[0]*im_shape[1])
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# train
im_torch = torch.from_numpy(pil_to_np(im)).unsqueeze(0).to(device)

model = MyNet(im_torch.size(1)).to(device)
model.apply(weights_init)
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
label_colours = np.random.randint(255, size=(args.nChannel, 3))

if args.warmup:

    # warmup
    warmup_steps = args.wuSteps
    warmup_lr = args.lr
    for step, batch_idx in enumerate(range(warmup_steps), 1):

        optimizer = optim.SGD(model.parameters(), lr=step * (warmup_lr / warmup_steps))

        loss, nLabels = train_iter(model, optimizer, loss_fn, label_colours)

        print(batch_idx, '/', warmup_steps, ':', nLabels, loss.item())

        writer.add_scalar('train/warmup_loss', loss.item(), batch_idx)
        writer.add_image('train/warmup_seg', create_segmentation(model, im_torch, im_shape, label_colours, args)[0],
                         batch_idx, dataformats='HWC')


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
if args.cosine_sched:
    T_MAX = 2
    T_MULT = 2
    steps = 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX)
else:
    scheduler = None

count = None
for batch_idx in range(args.maxIter):

    if scheduler is not None:
        if steps == (T_MAX - 1):
            T_MAX *= T_MULT
            steps = 0
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX, last_epoch=0)
        else:
            scheduler.step()
            steps += 1

    loss, nLabels = train_iter(model, optimizer, loss_fn, label_colours)

    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

    writer.add_scalar('train/loss', loss.item(), batch_idx)
    writer.add_image('train/seg', create_segmentation(model, im_torch, im_shape, label_colours, args)[0],
                     batch_idx, dataformats='HWC')

    # if we reached min labels we train for additional 50 iterations for stabilization
    if count is None:
        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            count = 50
    elif count > 0 and nLabels >= args.minLabels:
        count -= 1
    else:
        break

# save output image
im_target_rgb, im_target = create_segmentation(model, im_torch, im_shape, label_colours, args)
im_target_rgb = cv2.cvtColor(im_target_rgb, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(run_dir, args.output), im_target_rgb)

writer.close()
