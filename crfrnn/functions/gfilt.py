import torch as th
import numpy as np

from permutohedral import gfilt_cuda
from .hash import Hashtable

def make_gfilt_buffers(val_dim, h, w, cap, dev):
    return [th.zeros(val_dim, h, w, device=dev),   # output
            th.zeros(cap, val_dim, device=dev),    # tmp_vals_1
            th.zeros(cap, val_dim, device=dev)]    # tmp_vals_2

def make_gfilt_hash(ref):
    ref_dim = ref.shape[0]
    return Hashtable.forward(None, ref * float(np.sqrt(2/3) * (ref_dim+1)))

class GaussianFilter(th.autograd.Function):
    @staticmethod
    def forward(ctx, ref, val, _hash_buffers=None, _gfilt_buffers=None):
        if not val.is_contiguous():
            val = val.contiguous()

        ref_dim = ref.shape[0]
        val_dim, h, w = val.shape

        if _hash_buffers is None:
            hash_buffers = make_gfilt_hash(ref)
        else:
            hash_buffers = list(_hash_buffers)

        cap = hash_buffers[0].shape[0]

        if _gfilt_buffers is None:
            gfilt_buffers = make_gfilt_buffers(val_dim, h, w, cap, val.device)
        else:
            gfilt_buffers = list(_gfilt_buffers)

        args = [val] + gfilt_buffers +  hash_buffers + [cap, ref_dim, False]
        if val.is_cuda:
            gfilt_cuda(*args)
        else:
            raise NotImplementedError("Gfilt currently requires CUDA support.")
            #gfilt_cpu(*args)

        out = gfilt_buffers[0].clone()

        if ref.requires_grad:
            ctx.save_for_backward(ref, val, out, *hash_buffers)
        elif val.requires_grad:
            ctx.save_for_backward(ref, val, *hash_buffers)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grads = [None, None, None, None]

        ref = ctx.saved_tensors[0]
        val = ctx.saved_tensors[1]
        hash_buffers = list(ctx.saved_tensors[-6:])

        ref_dim = ref.shape[0]
        h, w = val.shape[-2:]
        cap = hash_buffers[0].shape[0]

        def filt(v):
            if not v.is_contiguous():
                v = v.contiguous()
            gfilt_buffers = make_gfilt_buffers(v.shape[0], h, w, cap, v.device)
            args = [v] + gfilt_buffers +  hash_buffers + [cap, ref_dim, True]
            gfilt_cuda(*args)
            return gfilt_buffers[0]

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            filt_og = filt(grad_output)

        if ctx.needs_input_grad[0]:
            out = ctx.saved_tensors[2]

            grads[0] = th.stack(
                [
                    (grad_output * (filt(val * r_i) - r_i * out)) +
                    (val * (filt(grad_output * r_i) - r_i * filt_og))
                    for r_i in ref
                ]
            ).sum(dim=1)

        if ctx.needs_input_grad[1]:
            grads[1] = filt_og

        return grads[0], grads[1], grads[2], grads[3]

gfilt = GaussianFilter.apply
