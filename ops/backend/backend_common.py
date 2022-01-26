import torch
import numpy as np

def maskns(J, M, N):
    m = torch.ones(J, M, N)
    for j in range(J):
        for x in range(M):
            for y in range(N):
                if (x<(2**j)//2 or y<(2**j)//2 \
                or x+1>M-(2**j)//2 or y+1>N-(2**j)//2):
                    m[j, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m

def masks_subsample_shift(J,M,N):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            for j_ in range(j,J):
                m[j_,x,y] = 1.
    return m


class SubInitSpatialMean(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -1, True)
            minput = torch.mean(minput, -2, True)
            self.minput = minput
        output = input - self.minput
        return output


class DivInitStd(object):
    def __init__(self,stdcut=1e-9):
        self.stdinput = None
        self.eps = stdcut

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(...,M,N)
            m = torch.mean(torch.mean(stdinput, -1, True), -2, True)
            stdinput = stdinput - m
            d = input.shape[-1]*input.shape[-2]
            stdinput = torch.norm(stdinput, dim=(-2,-1), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps

        output = input/self.stdinput
        return output


def padc(x):
    x_ = x.clone()
    return torch.stack((x_, torch.zeros_like(x_)), dim=-1)
