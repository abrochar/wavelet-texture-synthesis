import torch
import torch.nn as nn
import numpy as np
import scipy.io
import os
import sys
sys.path.append(os.getcwd())
from PIL import Image
import skimage.transform as sk

def load_image_gray(name, size=256):

    im_ = Image.open('./images/'+name+'.jpg')
    im = np.array(im_)/255.
    if im.shape[-1] == 3:
        im = im.mean(axis=-1)

    if im.shape[-1] != size:
        im = sk.resize(im, (size, size), preserve_range=True, anti_aliasing=True)
    im = torch.tensor(im).type(torch.float).unsqueeze(0).unsqueeze(0).cuda()

    return im


def load_image_color(name, size=256):

    im = Image.open('./images/'+name+'.jpg')
    im = np.array(im)/255.
    im = sk.resize(im, (size, size, 3), preserve_range=True, anti_aliasing=True)
    im = torch.tensor(im).type(torch.float)
    im = im.permute(2,0,1).unsqueeze(0).cuda()

    return im
