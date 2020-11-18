import torch as th
import torch.nn as nn
import torch.nn.functional as thf

from permutohedral.gfilt import gfilt


def gaussian_filter(ref, val, kstd):
    return gfilt(ref / kstd[:, :, None, None], val)


def mgrid(h, w, dev):
    y = th.arange(0, h, device=dev).repeat(w, 1).t()
    x = th.arange(0, w, device=dev).repeat(h, 1)
    return th.stack([y, x], 0)


def gkern(std, chans, dev):
    sig_sq = std ** 2
    r = sig_sq if (sig_sq % 2) else sig_sq - 1
    s = 2 * r + 1
    k = th.exp(-((mgrid(s, s, dev) - r) ** 2).sum(0) / (2 * sig_sq))
    W = th.zeros(chans, chans, s, s, device=dev)
    for i in range(chans):
        W[i, i] = k / k.sum()
    return W


class CRF(nn.Module):
    def __init__(
        self,
        n_ref: int,
        n_out: int,
        sxy_bf: float = 70,
        sc_bf: float = 12,
        compat_bf: float = 4,
        sxy_spatial: float = 6,
        compat_spatial: float = 2,
        num_iter: int = 5,
        normalize_final_iter: bool = True,
        trainable_kstd: bool = False,
    ):
        """Implements fast approximate mean-field inference for a
        fully-connected CRF with Gaussian edge potentials within a neural
        network layer using fast bilateral filtering.

        Args:
            n_ref: Number of channels in the reference images.

            n_out: Number of labels.

            sxy_bf: Spatial standard deviation of the bilateral filter.

            sc_bf: Color standard deviation of the bilateral filter.

            compat_bf: Label compatibility weight for the bilateral filter.
                Assumes a Potts model w/one parameter.

            sxy_spatial: Spatial standard deviation of the 2D Gaussian
                convolution kernel.

            compat_spatial: Label compatibility weight of the 2D Gaussian
                convolution kernel.

            num_iter: Number of steps to run in the inference loop.

            normalize_final_iter: If pre-softmax outputs are desired rather
                than label probabilities, set this to False.

            trainable_kstd: Allow the parameters of the bilateral filter to be
                learned as well. This option may make training less stable.
        """
        assert n_ref in {1, 3}, "Reference image must be either RGB or greyscale (3 or 1 channels)."

        super().__init__()

        self.n_ref = n_ref
        self.n_out = n_out
        self.sxy_bf = sxy_bf
        self.sc_bf = sc_bf
        self.compat_bf = compat_bf
        self.sxy_spatial = sxy_spatial
        self.compat_spatial = compat_spatial
        self.num_iter = num_iter
        self.normalize_final_iter = normalize_final_iter
        self.trainable_kstd = trainable_kstd

        kstd = th.FloatTensor([sxy_bf, sxy_bf, sc_bf, sc_bf, sc_bf])
        if n_ref == 1:
            kstd = kstd[:3]

        if trainable_kstd:
            self.kstd = nn.Parameter(kstd)
        else:
            self.register_buffer("kstd", kstd)

        self.register_buffer("gk", gkern(sxy_spatial, n_out))

    def forward(self, unary, ref):
        def _bilateral(V, R):
            return gaussian_filter(R, V, self.kstd[None])

        def _step(prev_q, U, ref, normalize=True):
            qbf = _bilateral(prev_q, ref)
            qsf = thf.conv2d(prev_q, self.gk, padding=self.gk.shape[-1] // 2)
            q_hat = -self.compat_bf * qbf - self.compat_spatial * qsf
            q_hat = U - q_hat
            return th.softmax(q_hat, dim=1) if normalize else q_hat

        def _inference(unary, ref):
            U = th.log(th.clamp(unary, 1e-5, 1))
            prev_q = th.softmax(U, dim=1)

            for i in range(self.num_iter):
                normalize = self.normalize_final_iter or i < self.num_iter - 1
                prev_q = _step(prev_q, U, ref, normalize=normalize)
            return prev_q

        N, _, H, W = unary.shape
        yx = mgrid(H, W, unary.device)
        grid = yx[None].repeat(N, 1, 1, 1)
        stacked = th.cat([grid, ref], dim=1)

        return _inference(unary, stacked)
