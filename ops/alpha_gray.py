import torch
import torch.fft as fft
import math
import numpy as np
import scipy.io as sio
import math
from .backend import SubInitSpatialMean, DivInitStd, \
    padc, maskns, masks_subsample_shift

class ALPHA(object):

    def __init__(self, M=256, N=256, J=5, L=4, A=4, A_prime=1,
                 delta_j=4,
                 nb_chunks=1, chunk_id=0,
                 shift='all', wavelets='morlet'):
        self.M, self.N, self.J, self.L = M, N, J, L
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.A = A
        self.A_prime = A_prime
        self.delta_j = delta_j
        self.wavelets = wavelets
        self.shift = shift
        assert(self.chunk_id <= self.nb_chunks)
        self.build()

    def build(self):

        self.masks_shift = masks_subsample_shift(self.J,self.M,self.N)
        self.masks_shift = torch.cat((torch.zeros(1,self.M, self.N),
                                      self.masks_shift), dim=0)
        self.masks_shift[0,0,0] = 1.
        self.factr_shift = self.masks_shift.sum(dim=(-2,-1))

        self.filters_tensor()

        self.idx_wph = self.compute_idx()
        self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)

        # define variables to store mean and std of observation
        self.subinitmean1 = SubInitSpatialMean()
        self.subinitmean2 = SubInitSpatialMean()
        self.divinitstd1 = DivInitStd()
        self.divinitstd2 = DivInitStd()

        self.divinitstdJ = DivInitStd()
        self.subinitmeanJ = SubInitSpatialMean()

        if self.wavelets == 'morlet':
            self.divinitstdH = [None,None,None]
            for hid in range(3):
                self.divinitstdH[hid] = DivInitStd()
        if self.wavelets == 'steer':
            self.subinitmean0 = SubInitSpatialMean()
            self.divinitstd0 = DivInitStd()


    def filters_tensor(self):

        J = self.J
        M = self.M; N = self.N; L = self.L

        if self.wavelets == 'morlet':
            hatpsi = torch.load('./filters/morlet_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt')  # (J,L,M,N)
            hatphi = torch.load('./filters/morlet_lp_N'+str(N)+'_J'+str(J)+'_L'+str(L)+'.pt')  # (M,N)

        A = self.A
        A_prime = self.A_prime

        alphas_ = torch.arange(A, dtype=torch.float)/A*2*math.pi
        alphas = torch.complex(torch.cos(alphas_),
                               torch.sin(alphas_))

        filt = torch.zeros(J, L, A, M, N, dtype=torch.cfloat)
        for alpha in range(A):
            for j in range(J):
                for theta in range(L):
                    psi_signal = hatpsi[j, theta, ...]
                    filt[j, theta, alpha, :, :] = alphas[alpha]*psi_signal

        self.hatphi = hatphi
        self.hatpsi = filt

        # add haar filters for high frequencies
        self.hathaar2d = torch.zeros(3, M, N, dtype=torch.cfloat)
        psi = torch.zeros(M,N,2)
        psi[1,1,1] = 1/4
        psi[1,2,1] = -1/4
        psi[2,1,1] = 1/4
        psi[2,2,1] = -1/4
        self.hathaar2d[0,:,:] = fft.fft2(torch.view_as_complex(psi))

        psi[1,1,1] = 1/4
        psi[1,2,1] = 1/4
        psi[2,1,1] = -1/4
        psi[2,2,1] = -1/4
        self.hathaar2d[1,:,:] = fft.fft2(torch.view_as_complex(psi))

        psi[1,1,1] = 1/4
        psi[1,2,1] = -1/4
        psi[2,1,1] = -1/4
        psi[2,2,1] = 1/4
        self.hathaar2d[2,:,:] = fft.fft2(torch.view_as_complex(psi))

        # load masks for aperiodicity
        self.masks = maskns(J, M, N).unsqueeze(1).unsqueeze(1)  # (J, M, N)


    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['la1'])
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks-1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk*(nb_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['shifted'] = self.idx_wph['shifted'][offset:offset+nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]

#        print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)

        return this_wph


    def to_shift(self, j1, j2, l1, l2):
        if self.shift == 'all':
            return True
        elif self.shift == 'same':
            return (j1 == j2) and (l1 ==l2)


    def compute_idx(self):
        L = self.L
        J = self.J
        A = self.A
        A_prime = self.A_prime
        dj = self.delta_j

        idx_la1 = []
        idx_la2 = []
        shifted = []
        nb_moments = 0

        for j1 in range(J):
            for j2 in range(j1, min(J,j1+1+dj)):
                for l1 in range(L):
                    for l2 in range(L):
                        for alpha1 in range(A):
                            for alpha2 in range(A_prime):
                                if self.to_shift(j1, j2, l1, l2):  # coeffs whith shits
                                    idx_la1.append(A*L*j1+A*l1+alpha1)
                                    idx_la2.append(A*L*j2
                                                   + A*l2+alpha2)
                                    shifted.append(J)
                                    nb_moments += int(self.factr_shift[-1])
                                else:
                                    idx_la1.append(A*L*j1+A*l1+alpha1)
                                    idx_la2.append(A_prime*L*j2
                                                   + A_prime*l2+alpha2)
                                    shifted.append(0)
                                    nb_moments += 1

        print('number of moments (without low-pass and harr): ', nb_moments)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['shifted'] = torch.tensor(shifted).type(torch.long)

        return idx_wph


    def cuda(self):
        """
            Moves the parameters of the scattering to the GPU
        """
        self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor)
        self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor)

        self.hatpsi = self.hatpsi.cuda()
        self.hatphi = self.hatphi.cuda()
        if self.wavelets == 'morlet':
            self.hathaar2d = self.hathaar2d.cuda()
        self.masks = self.masks.cuda()
        self.masks_shift = self.masks_shift.cuda()

        return self

    def cpu(self):
        """
            Moves the parameters of the scattering to the CPU
        """
        return self._type(torch.FloatTensor)

    def forward(self, input):

        J = self.J
        L = self.L
        M = self.M
        N = self.N
        A = self.A
        A_prime = self.A_prime
        phi = self.hatphi
        wavelets = self.wavelets

        x_c = padc(input)  # add zeros for imaginary part
        x_c = torch.view_as_complex(x_c)
        hatx_c = fft.fft2(x_c)

        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatpsi_la = self.hatpsi  # (J,L,A,M,N)
        assert(nb == 1 and nc == 1)
        nb_channels = self.this_wph['la1'].shape[0]
        t = 3 if wavelets == 'morlet' else 1 if wavelets == 'steer' else 0
        if self.chunk_id < self.nb_chunks-1:
            Sout = input.new(nb, nc, nb_channels,M,N)
        else:
            Sout = input.new(nb, nc,
                             nb_channels+1+t,
                             M,N)
        hatx_bc = hatx_c[0, 0, ...]  # (M,N)
        # convolution in Fourier domain
        hatxpsi_bc = hatpsi_la * hatx_bc.view(1,1,1,M,N)  # (J,L,A,M,N)
        xpsi_bc = fft.ifft2(hatxpsi_bc)  # (J,L,A,M,N)
        # apply non linearity
        xpsi_bc_ = torch.real(xpsi_bc).relu()  # (J,L,A,M,N)
        # masks for non periodic images
        xpsi_bc_ = xpsi_bc_ * self.masks  # (J,L,A,M,N)
        # renorm by observation stats
        xpsi_bc0 = self.subinitmean1(xpsi_bc_)
        xpsi_bc0_n = self.divinitstd1(xpsi_bc0)
        # reshape and select indices
        xpsi_bc0_ = xpsi_bc0_n.view(1, J*L*A, M, N)  # (1,J*L*A,M,N)
        xpsi_bc_la1 = xpsi_bc0_[:,self.this_wph['la1'],...]   # (1,P_c,M,N)
        xpsi_bc_la2 = xpsi_bc0_[:,self.this_wph['la2'],...] # (1,P_c,M,N)
        # shifted correlations in Fourier domain
        x1 = torch.view_as_complex(padc(xpsi_bc_la1))
        x2 = torch.view_as_complex(padc(xpsi_bc_la2))
        hatconv_xpsi_bc = fft.fft2(x1) * torch.conj(fft.fft2(x2))
        conv_xpsi_bc = torch.real(fft.ifft2(hatconv_xpsi_bc))
        # select shifted coefficients
        masks_shift = self.masks_shift[self.this_wph['shifted'],...].view(1,-1,M,N)
        corr_bc = conv_xpsi_bc * masks_shift

        Sout[0, 0, 0:nb_channels,...] = corr_bc[0, ...]

        # low-pass and high-pass stats
        if self.chunk_id == self.nb_chunks-1:
            # low-pass
            hatxphi_c = hatx_c * self.hatphi.view(1,M,N)  # (nb,nc,M,N)
            xphi_c = fft.ifft2(hatxphi_c)
            xphi0_c = self.subinitmeanJ(xphi_c)
            xphi0_c = self.divinitstdJ(xphi0_c)
            xphi0_c = xphi0_c * self.masks[-1,...].view(1,1,M,N)
            xphi0_c = torch.view_as_complex(padc(xphi0_c.abs()))
            xphi0_c = fft.fft2(xphi0_c)
            xphi0_c = fft.ifft2(xphi0_c * torch.conj(xphi0_c))
            xphi0_mean = torch.real(xphi0_c) * self.masks_shift[-1,...].view(1,1,M,N)
            Sout[0, 0, -1,...] = xphi0_mean[0, 0, ...]

            # haar filters for morlet wavelets
            if wavelets == 'morlet':
                for hid in range(3):
                    hatxpsih_c = hatx_c * self.hathaar2d[hid,:,:].view(1,M,N) # (nb,nc,M,N)
                    xpsih = fft.ifft2(hatxpsih_c)
                    xpsih = self.divinitstdH[hid](xpsih)
                    xpsih = torch.abs(xpsih)
                    xpsih = xpsih * self.masks[0,...].view(1,1,M,N)
                    # shifted correlations in Fourier domain
                    xpsih = torch.view_as_complex(padc(xpsih))
                    xpsih = fft.fft2(xpsih)
                    xpsih = fft.ifft2(xpsih * torch.conj(xpsih))
                    xpsih = torch.real(xpsih[0,...]) * self.masks_shift[-1,...]
                    Sout[0,0,-4+hid,...] = xpsih

            # psi0 for steer
            if wavelets == 'steer':
                hatxpsi0 = hatx_c * self.hatpsi0  # (1,1,M,N)
                xpsi0 = fft.ifft2(hatxpsi0)
                xpsi0 = self.subinitmean0(xpsi0)
                xpsi0 = self.divinitstd0(xpsi0)
                xpsi0 = xpsi0 * self.masks[0,...].view(1,1,M,N)
                xpsi0 = torch.view_as_complex(padc(xpsi0.abs()))
                xpsi0 = fft.fft2(xpsi0)
                xpsi0 = fft.ifft2(xpsi0 * torch.conj(xpsi0))
                xpsi0 = torch.real(xpsi0[0,...]) * self.masks_shift[-1,...]
                Sout[0, 0, -2,...] = xpsi0

        return Sout

    def __call__(self, input):
        return self.forward(input)
