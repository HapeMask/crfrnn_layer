import torch as th
import torch.nn as nn
import numpy as np

from permutohedral.hash import get_hash_cap, make_hashtable
from permutohedral.gfilt import gfilt, make_gfilt_buffers

def gaussian_filter(ref, val, kstd, hb=None, gb=None):
    return gfilt(ref / kstd[:, None, None], val, hb, gb)

def mgrid(h, w, dev):
    y = th.arange(0, h, device=dev).repeat(w, 1).t()
    x = th.arange(0, w, device=dev).repeat(h, 1)
    return th.stack([y, x], 0)

def gkern(std, chans, dev):
    sig_sq = std ** 2
    r = sig_sq if (sig_sq % 2) else sig_sq - 1
    s = 2 * r + 1
    k = th.exp(-((mgrid(s, s, dev)-r) ** 2).sum(0) / (2 * sig_sq))
    W = th.zeros(chans, chans, s, s, device=dev)
    for i in range(chans):
        W[i, i] = k / k.sum()
    return W

class CRF(nn.Module):
    def __init__(self, sxy_bf=70, sc_bf=12, compat_bf=4, sxy_spatial=6,
                 compat_spatial=2, num_iter=5, normalize_final_iter=True,
                 trainable_kstd=False):

        super().__init__()

        self.sxy_bf = sxy_bf
        self.sc_bf = sc_bf
        self.compat_bf = compat_bf
        self.sxy_spatial = sxy_spatial
        self.compat_spatial = compat_spatial
        self.num_iter = num_iter
        self.normalize_final_iter = normalize_final_iter
        self.trainable_kstd = trainable_kstd

        if isinstance(sc_bf, (int, float)):
            sc_bf = 3 * [sc_bf]

        kstd = th.FloatTensor([sxy_bf, sxy_bf, sc_bf[0], sc_bf[1], sc_bf[2]])
        if trainable_kstd:
            self.kstd = nn.Parameter(kstd)
        else:
            self.register_buffer("kstd", kstd)

    def forward(self, unary, ref):
        N, ref_dim, H, W = ref.shape
        Nu, val_dim, Hu, Wu = unary.shape
        assert(Nu == N and Hu == H and Wu == W)

        if ref_dim not in [3, 1]:
            raise ValueError("Reference image must be either color or greyscale (3 or 1 channels).")
        ref_dim += 2

        kstd = self.kstd[:3] if ref_dim == 3 else self.kstd
        gk = gkern(self.sxy_spatial, val_dim, unary.device)

        yx = mgrid(H, W, unary.device)
        grid = yx[None].repeat(N, 1, 1, 1)

        cap = get_hash_cap(H * W, ref_dim)
        stacked = th.cat([grid, ref], dim=1)
        gb = make_gfilt_buffers(val_dim, H, W, cap, unary.device)
        gb1 = make_gfilt_buffers(1, H, W, cap, unary.device)

        def _bilateral(V, R, hb):
            o = th.ones(1, H, W, device=unary.device)
            norm = th.sqrt(gaussian_filter(R, o, kstd, hb, gb1)) + 1e-8
            return gaussian_filter(R, V / norm, kstd, hb, gb) / norm

        def _step(prev_q, U, ref, hb, normalize=True):
            qbf = _bilateral(prev_q, ref, hb)
            qsf = th.nn.functional.conv2d(prev_q[None], gk, padding=gk.shape[-1]//2)[0]

            q_hat = -self.compat_bf * qbf - self.compat_spatial * qsf
            q_hat = U - q_hat

            return th.softmax(q_hat, dim=0) if normalize else q_hat

        def _inference(unary_i, ref_i):
            U = th.log(th.clamp(unary_i, 1e-5, 1))
            prev_q = th.softmax(U, dim=0)
            hb = make_hashtable(ref_i.data / kstd[:, None, None].data)

            for i in range(self.num_iter):
                normalize = self.normalize_final_iter or i < self.num_iter - 1
                prev_q = _step(prev_q, U, ref_i, hb, normalize=normalize)
            return prev_q

        return th.stack([_inference(unary[i], stacked[i]) for i in range(N)])
