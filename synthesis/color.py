import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
from load_image import load_image_color
import sys
import math
import argparse
sys.path.append(os.getcwd())
from routine_color import *
from hist import *

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)

gpu = True

# Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--N', type=int, default=256)
parser.add_argument('--image', default='gravel')
parser.add_argument('--model', default='alpha')
parser.add_argument('--J', type=int, default=5)
parser.add_argument('--L', type=int, default=4)
parser.add_argument('--dj', type=int, default=4)
parser.add_argument('--A', type=int, default=4)
parser.add_argument('--A_prime', type=int, default=1)
parser.add_argument('--wavelets', default='morlet')
parser.add_argument('--shift', default='samec') # 'samec' or 'all'
parser.add_argument('--nb_chunks', type=int, default=11)
parser.add_argument('--nb_restarts', type=int, default=1)
parser.add_argument('--nGPU', type=int, default=1)
parser.add_argument('--maxite', type=int, default=500)
parser.add_argument('--factr', type=int, default=1e-3)
parser.add_argument('--nb_syn', type=int, default=1)
parser.add_argument('--hist', action='store_false')
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_false')
args = parser.parse_args()


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

from ops.alpha_color import ALPHA

# load image
im = load_image_color(args.image, args.N)
M, N = im.size(-1), im.size(-2)
mean_ = im.mean(dim=(-2,-1), keepdim=True)
std_ = im.std(dim=(-2,-1), keepdim=True)

# plot input image
#plt.imshow(im.cpu().squeeze().permute(1,2,0))
#plt.show()


# get color correlation mat
C = get_color_corr(im)

# synthesis
J = args.J
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
                       shift,
                       nb_chk, chk_id,
                       wavelets, devid)
        wph_op = wph_op.cuda()
        wph_ops[chk_id] = wph_op
        im_dev = im.to(devid)
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            Sim_ = wph_op(im_dev)
            opid += 1
            Sims.append(Sim_)
    torch.cuda.synchronize()

    x0 = torch.normal(mean_.repeat(1,1,M,N), std_.repeat(1,1,M,N))

    # run optim
    x_fin = call_lbfgs2_routine(x0,0,wph_ops,wph_streams,Sims,C,
                                nb_restarts,maxite,factr,nGPU)


# convert synthesis to numpy
im_opt = x_fin.detach().cpu().squeeze().numpy()

# reshape to (M,N,3)
im_opt = np.moveaxis(im_opt, 0, -1)
im = im.squeeze().permute(1,2,0).cpu().numpy()
# match histogram
if args.hist:
    im_opt = histogram_matching(im_opt, im, grey=False)

# plot synthesis
if args.plot:
    plt.imshow(im_opt, vmin=0, vmax=1)
    plt.show()
if args.save:
    if not os.path.exists('./results'):
        os.mkdir('./results')
    name = args.image + '_color.npy'
    np.save('./results/'+name, im_opt)
