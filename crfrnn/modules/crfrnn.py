import torch as th
import torch.nn as nn
import numpy as np

from ..functions import gfilt
from ..functions.hash import Hashtable, get_hash_cap
from ..functions.gfilt import make_gfilt_buffers, make_gfilt_hash

def gaussian_filter(ref, val, kstd, hb=None, gb=None):
    return gfilt(ref / kstd[:, None, None], val, hb, gb)

def mgrid(h, w):
    y = th.arange(0, h).repeat(w, 1).t()
    x = th.arange(0, w).repeat(h, 1)
    return th.stack([y, x], 0)

def gkern(std, chans):
    sig_sq = std**2
    r = sig_sq if (sig_sq % 2) else sig_sq-1
    s = 2*r + 1
    k = th.exp(-((mgrid(s, s)-r)**2).sum(0) / (2*sig_sq))
    W = th.zeros((chans, chans, s, s))
    for i in range(chans):
        W[i, i] = k / k.sum()
    return W

def softmax(x, dim=1):
    e_x = th.exp(x - x.max(dim=dim, keepdim=True)[0])
    return e_x / e_x.sum(dim=dim, keepdim=True)

class CRF(nn.Module):
    def __init__(self, sxy_bf=70, sc_bf=10, compat_bf=6, sxy_spatial=2,
                 compat_spatial=2, num_iter=5, normalize_final_iter=True):

        super(CRF, self).__init__()
        self.sxy_bf = sxy_bf
        self.sc_bf = sc_bf
        self.compat_bf = compat_bf
        self.sxy_spatial = sxy_spatial
        self.compat_spatial = compat_spatial
        self.num_iter = num_iter
        self.normalize_final_iter = normalize_final_iter

        self.kstd = nn.Parameter(th.FloatTensor([sxy_bf, sxy_bf,
                                                 sc_bf, sc_bf, sc_bf]))

    def forward(self, unary, ref):
        cuda = self.kstd.is_cuda
        N, ref_dim, H, W = ref.shape
        Nu, val_dim, Hu, Wu = unary.shape
        assert(Nu == N and Hu == H and Wu == W)

        if not isinstance(unary, th.autograd.Variable):
            unary = th.autograd.Variable(unary)
        if not isinstance(ref, th.autograd.Variable):
            ref = th.autograd.Variable(ref)

        if ref_dim not in [3, 1]:
            raise ValueError("Reference image must be either color or greyscale \
(3 or 1 channels).")
        ref_dim += 2

        kstd = self.kstd[:3] if ref_dim == 3 else self.kstd
        gk = th.autograd.Variable(gkern(self.sxy_spatial, val_dim))

        yx = mgrid(H, W)
        grid = yx[None].repeat(N, 1, 1, 1)
        grid = th.autograd.Variable(grid, requires_grad=False)

        if cuda:
            gk, grid = [v.cuda() for v in [gk, grid]]

        stacked = th.cat([grid, ref], dim=1)
        gb = make_gfilt_buffers(val_dim, H, W, get_hash_cap(H*W, ref_dim), cuda)
        gb1 = make_gfilt_buffers(1, H, W, get_hash_cap(H*W, ref_dim), cuda)

        def _bilateral(V, R, hb):
            o = th.ones(1, H, W)
            if cuda:
                o = o.cuda()
            norm = th.sqrt(gaussian_filter(R, o, kstd, hb, gb1)) + 1e-8
            return gaussian_filter(R, V/norm, kstd, hb, gb) / norm

        def _step(prev_q, U, ref, hb, normalize=True):
            qbf = _bilateral(prev_q, ref, hb)
            qsf = th.nn.functional.conv2d(prev_q[None], gk, padding=gk.shape[-1]//2)[0]

            q_hat = -self.compat_bf * qbf + -self.compat_spatial * qsf
            q_hat = U - q_hat

            return softmax(q_hat, dim=0) if normalize else q_hat

        def _inference(unary_i, ref_i):
            U = th.log(th.clamp(unary_i, 1e-5, 1))
            prev_q = softmax(U, dim=0)
            hb = make_gfilt_hash(ref_i.data / kstd[:, None, None].data)

            for i in range(self.num_iter):
                normalize = self.normalize_final_iter or i < self.num_iter-1
                prev_q = _step(prev_q, U, ref_i, hb, normalize=normalize)
            return prev_q

        return th.stack([_inference(unary[i], stacked[i]) for i in range(N)])
