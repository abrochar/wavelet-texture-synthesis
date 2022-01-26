import os
import matplotlib.pyplot as plt
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

torch.backends.cudnn.deterministic = True


def obj_func_id(im_t,j,wph_ops,wph_streams,Sims,factr2,op_id,nGPU):
    # convert points to im on devid, using a loop
    devid = op_id % nGPU
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
        # compute wph grad on devid
        wph_op = wph_ops[op_id]
        im_t_j = torch.nn.AvgPool2d(2**j)(im_t)
        p = wph_op(im_t_j)
#        diff = p/(Sims[op_id]+1e-9)-1
        diff = p - Sims[op_id]
        loss = torch.mul(diff,diff).sum()
        loss = loss*factr2

    return loss


def obj_func_id_1gpu(im_t,j,wph_ops,wph_streams,Sims,factr2,op_id):
    wph_op = wph_ops[op_id]
    if im_t.shape[-1]==3:
      im_t_ = im_t[...,0]*0.2125 + im_t[...,1]*0.7154 + im_t[...,2]*0.0721
      im_t_j = torch.nn.AvgPool2d(2**j)(im_t_)
    else:
        im_t_j = torch.nn.AvgPool2d(2**j)(im_t)
    p = wph_op(im_t_j)
    diff = p - Sims[op_id]
    loss = torch.mul(diff,diff).sum()
    loss = loss*factr2

    return loss


def obj_func(x,j,wph_ops,wph_streams,Sims,factr2,nGPU):
    loss = 0
    loss_a = []

    # copy x to multiple gpus
    x_a = []
    for devid in range(nGPU):
        x_t = x.to(devid)
        x_a.append(x_t)

    # compute gradients with respect to x_a
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        x_t = x_a[devid]
        loss_t = obj_func_id(x_t,j,wph_ops,wph_streams,Sims,factr2,op_id,nGPU)
        loss_t.backward(retain_graph=False) # accumulate grad into x.grad
        loss_a.append(loss_t)

    torch.cuda.synchronize()

    # sum the loss
    for op_id in range(len(wph_ops)):
        loss = loss + loss_a[op_id].item()

    return loss


def obj_func_1gpu(x,j,wph_ops,wph_streams,Sims,factr2):
    loss = 0
    if x.grad is not None:
        x.grad.data.zero_()
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id_1gpu(x,j,wph_ops,wph_streams,Sims,factr2,op_id)
        loss_t.backward() # accumulate grad into x.grad
        loss = loss + loss_t
    return loss


def call_lbfgs2_routine(x0,j,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr, \
                        nGPU=2, maxcor=50,gtol=1e-10,ftol=1e-10):

    for start in range(nb_restarts+1):
        if start==0:
           x = x0
           x.requires_grad_(True)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)

        def closure():
#            pbar = tqdm(total = maxite)
            optimizer.zero_grad()
            if nGPU>=2:
                loss = obj_func(x,j,wph_ops,wph_streams,Sims,factr**2,nGPU)
            else:
                loss = obj_func_1gpu(x,j,wph_ops,wph_streams,Sims,factr**2)
            pbar.update(1)

            return loss
        pbar = tqdm(total = maxite)
        optimizer.step(closure)
        pbar.close()


        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        final_loss = opt_state['prev_loss']
        print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
        
    return x
