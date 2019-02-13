import torch as th
import numpy as np

from .._ext.crfrnn import gfilt_cuda
from .hash import Hashtable

def make_gfilt_buffers(val_dim, h, w, cap, cuda=False, dev=None):
    buffers = [th.zeros(val_dim, h, w),   # output
               th.zeros(cap, val_dim),    # tmp_vals_1
               th.zeros(cap, val_dim)]    # tmp_vals_2
    if cuda:
        buffers = [b.cuda(dev) for b in buffers]
    return buffers

def make_gfilt_hash(ref):
    ref_dim = ref.shape[0]
    return Hashtable.forward(None, ref * float(np.sqrt(2/3) * (ref_dim+1)))

class GaussianFilter(th.autograd.Function):
    @staticmethod
    def forward(ctx, ref, val, _hash_buffers=None, _gfilt_buffers=None):
        is_cuda = val.is_cuda
        cudev = None
        if is_cuda:
            assert(ref.is_cuda and (val.get_device() == ref.get_device()))
            cudev = val.get_device()

        if not val.is_contiguous():
            val = val.contiguous()

        ref_dim = ref.shape[0]
        val_dim, h, w = val.shape
        N = h*w

        if _hash_buffers is None:
            hash_buffers = make_gfilt_hash(ref)
        else:
            hash_buffers = list(_hash_buffers)

        cap = hash_buffers[0].shape[0]

        if _gfilt_buffers is None:
            gfilt_buffers = make_gfilt_buffers(val_dim, h, w, cap, is_cuda, cudev)
        else:
            gfilt_buffers = list(_gfilt_buffers)

        nv = hash_buffers[-1][0]
        hash_buffers[-1] = nv

        # This isn't needed in python>=3.5
        args = [val] + gfilt_buffers +  hash_buffers + [cap, ref_dim]
        if val.is_cuda:
            gfilt_cuda(*args)
        else:
            raise NotImplementedError("Gfilt currently requires CUDA support.")
            #gfilt_cpu(*args)

        return gfilt_buffers[0]

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Not yet.")
        #grads = [None] * len(ctx.needs_input_grad)
        #if ctx.needs_input_grad[0]:
        #    grads[0] = compute_grad_r_HERE
        #if ctx.needs_input_grad[1]:
        #    grads[1] = compute_grad_v_HERE
        #return grads

gfilt = GaussianFilter.apply
