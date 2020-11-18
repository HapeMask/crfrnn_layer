import torch as th

from .hash import make_hashtable
import permutohedral_ext

th.ops.load_library(permutohedral_ext.__file__)
gfilt_cuda = th.ops.permutohedral_ext.gfilt_cuda


def make_gfilt_buffers(b, val_dim, h, w, cap, dev):
    return [
        th.zeros(b, val_dim, h, w, device=dev),  # output
        th.empty(cap, val_dim + 1, device=dev),  # tmp_vals_1
        th.empty(cap, val_dim + 1, device=dev),  # tmp_vals_2
    ]


class GaussianFilter(th.autograd.Function):
    @staticmethod
    def forward(ctx, ref, val, _hash_buffers=None, _gfilt_buffers=None):
        val = val.contiguous()
        b, ref_dim, h, w = ref.shape
        vb, val_dim, vh, vw = val.shape
        assert vb == b and vh == h and vw == w

        if _hash_buffers is None:
            hash_buffers = make_hashtable(ref)
        else:
            hash_buffers = list(_hash_buffers)
        hash_buffers[-1] = hash_buffers[-1].cpu()

        assert hash_buffers[0].shape[0] == b
        cap = hash_buffers[0].shape[1]

        if _gfilt_buffers is None:
            gfilt_buffers = make_gfilt_buffers(b, val_dim, h, w, cap, val.device)
        else:
            gfilt_buffers = list(_gfilt_buffers)

        if val.is_cuda:
            gfilt_cuda(val, *gfilt_buffers, *hash_buffers, ref_dim, False)
        else:
            raise NotImplementedError("Gfilt currently requires CUDA support.")
            # gfilt_cpu(val, *gfilt_buffers, *hash_buffers, cap, ref_dim, False)

        out = gfilt_buffers[0]

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

        b, ref_dim = ref.shape[:2]
        val_dim, h, w = val.shape[-3:]
        assert hash_buffers[0].shape[0] == b
        cap = hash_buffers[0].shape[1]

        def filt(v):
            if not v.is_contiguous():
                v = v.contiguous()
            gfilt_buffers = make_gfilt_buffers(b, val_dim, h, w, cap, v.device)
            gfilt_cuda(v, *gfilt_buffers, *hash_buffers, ref_dim, True)
            return gfilt_buffers[0]

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            filt_og = filt(grad_output)

        if ctx.needs_input_grad[0]:
            out = ctx.saved_tensors[2]

            grads[0] = th.stack(
                [
                    (grad_output * (filt(val * r_i) - r_i * out))
                    + (val * (filt(grad_output * r_i) - r_i * filt_og))
                    for r_i in ref.split(1, dim=1)
                ],
                dim=1,
            ).sum(dim=2)

        if ctx.needs_input_grad[1]:
            grads[1] = filt_og

        return grads[0], grads[1], grads[2], grads[3]


gfilt = GaussianFilter.apply
