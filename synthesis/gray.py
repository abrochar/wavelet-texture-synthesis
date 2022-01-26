import numpy as np
import torch
import matplotlib.pyplot as plt
from load_image import load_image_gray
import sys
import os
import math
import argparse
sys.path.append(os.getcwd())
from routine import call_lbfgs2_routine
from hist import *

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)

gpu = True

# Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--N', type=int, default=256)
parser.add_argument('-i', '--image', default='wood')
parser.add_argument('-m', '--model', default='alpha')
parser.add_argument('--J', type=int, default=5)
parser.add_argument('--L', type=int, default=4)
parser.add_argument('--dj', type=int, default=2)
parser.add_argument('--A', type=int, default=4)
parser.add_argument('-ap', '--A_prime', type=int, default=1)
parser.add_argument('--shift', default='all')
parser.add_argument('--wavelets', default='morlet')
parser.add_argument('--nb_chunks', type=int, default=3)
parser.add_argument('-r', '--nb_restarts', type=int, default=2)
parser.add_argument('--nGPU', type=int, default=1)
parser.add_argument('--maxite', type=int, default=500)
parser.add_argument('--factr', type=float, default=1e-3)
parser.add_argument('--nb_syn', type=int, default=1)
parser.add_argument('-h', '--hist', action='store_false')
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-p', '--plot', action='store_false')
args = parser.parse_args()


J = args.J
L = args.L
dj = args.dj
A = args.A
A_prime = args.A_prime
wavelets = args.wavelets
shift = args.shift
nb_chk = args.nb_chunks
nb_restarts = args.nb_restarts
nGPU = args.nGPU
maxite = args.maxite
factr = args.factr


from ops.alpha_gray import ALPHA


# load image
im = load_image_gray(args.image, args.N)
M, N = im.size(-1), im.size(-2)
v_min = np.percentile(im.cpu(), 1);
v_max = np.percentile(im.cpu(), 99)
mean_ = im.mean()
std_ = im.std()


# plot input image
#plt.imshow(im.cpu().squeeze())
#plt.show()

# synthesis
for syn in range(args.nb_syn):

    # for multi-GPU runs
    wph_streams = []
    for devid in range(nGPU):
        with torch.cuda.device(devid):
            s = torch.cuda.Stream()
            wph_streams.append(s)

    # compute descriptor for observation
    Sims = []
    opid = 0
    wph_ops = dict()
    for chk_id in range(nb_chk):
        devid = opid % nGPU
        wph_op = ALPHA(M, N, J, L, A, A_prime, dj,
                        nb_chk, chk_id, shift, wavelets)
        wph_op = wph_op.cuda()
        wph_ops[chk_id] = wph_op
        im_dev = im.to(devid)
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            Sim_ = wph_op(im_dev)
            opid += 1
            Sims.append(Sim_)
    torch.cuda.synchronize()

    # initialize with Gaussian whithe noise
    x0 = mean_ + torch.Tensor(1, 1, M, N).normal_(std=std_).cuda()

    # run optim
    x_fin = call_lbfgs2_routine(x0,0,wph_ops,wph_streams,Sims,
                                nb_restarts,maxite,factr,nGPU)
    x_opt = x_fin


# convert synthesis to numpy
im_opt = x_fin.detach().cpu().squeeze().numpy()
im = im.cpu().squeeze().numpy()
# match histogram
if args.hist:
    im_opt = histogram_matching(im_opt, im, grey=True)

# plot/save results

if args.plot:
    plt.imshow(im_opt, vmin=v_min, vmax=v_max, cmap='gray')
    plt.show()

if args.save:
    name = args.image + '_gray.npy'
    np.save('./results/'+name, im_opt)


