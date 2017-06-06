import numpy as np
import theano
import theano.tensor as tt
import lasagne.layers as ll

from .gpugfilt import gaussian_filter


class GaussianFilterLayer(ll.MergeLayer):
    def __init__(self, values, ref_img, kern_std, norm_type="sym",
                 name=None, trainable_kernels=False, _bilateral=False):

        assert(norm_type in ["sym", "pre", "post", None])
        super(GaussianFilterLayer, self).__init__(incomings=[values, ref_img],
                                                  name=name)

        self.val_dim = ll.get_output_shape(values)[1]
        self.ref_dim = ll.get_output_shape(ref_img)[1]

        if None in (self.val_dim, self.ref_dim):
            raise ValueError("Gaussian filtering requires known channel \
dimensions for all inputs.")

        self.norm_type = norm_type

        if _bilateral:
            self.ref_dim += 2

        if len(kern_std) != self.ref_dim:
            raise ValueError("Number of kernel weights must match reference \
dimensionality. Got %d weights for %d reference dims." % (len(kern_std),
                                                          self.ref_dim))

        self.kern_std = self.add_param(kern_std, (self.ref_dim,),
                                       name="kern_std",
                                       trainable=trainable_kernels,
                                       regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        vals, ref = inputs

        def filt(V, R):
            if self.norm_type is not None:
                o = tt.ones((1, V.shape[1], V.shape[2]), np.float32)
                norm = gaussian_filter(R, o, self.kern_std, self.ref_dim)
                norm = tt.sqrt(norm) if self.norm_type == "sym" else norm
                norm += 1e-8

            V = V / norm if self.norm_type in ["pre", "sym"] else V
            F = gaussian_filter(R, V, self.kern_std)
            return F / norm if self.norm_type in ["post", "sym"] else F

        filtered = theano.scan(fn=filt, sequences=[vals, ref],
                               outputs_info=None)[0]
        return filtered


class BilateralFilterLayer(GaussianFilterLayer):
    def __init__(self, values, ref_img, sxy=60, sc=10, norm_type="sym",
                 name=None):

        C = ll.get_output_shape(ref_img)[1]
        if C not in [1, 3]:
            raise ValueError("Bilateral filtering requires a color or \
greyscale reference image. Got %d channels." % C)

        if C == 1:
            kern_std = np.array([sxy, sxy, sc], np.float32)
        else:
            kern_std = np.array([sxy, sxy, sc, sc, sc], np.float32)

        super(BilateralFilterLayer, self).__init__(values, ref_img, kern_std,
                                                   norm_type, name=name,
                                                   _bilateral=True)

    def get_output_for(self, inputs, **kwargs):
        vals, ref = inputs
        N, _, H, W = ref.shape
        yx = tt.stack(tt.mgrid[0:H, 0:W])[np.newaxis, :, :, :]
        grid = tt.alloc(tt.cast(yx, "float32"), N, 2, H, W)
        stacked = tt.concatenate([grid, ref], axis=1)

        return super(BilateralFilterLayer, self).get_output_for(
                [vals, stacked], **kwargs)


def softmax(x, axis=1):
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def gkern(std, chans):
    sig_sq = std**2
    r = sig_sq if (sig_sq % 2) else sig_sq-1
    s = 2*r + 1
    k = np.exp(-((np.mgrid[:s, :s]-r)**2).sum(0) / (2*sig_sq))
    W = np.zeros((chans, chans, s, s), np.float32)
    W[np.arange(chans), np.arange(chans)] = k / k.sum()
    return W


class CRFasRNNLayer(ll.MergeLayer):
    """
    Layer implementing CRF inference as an unrolled RNN.

    The CRF inference process can be used to refine noisy per-pixel
    category predictions to be consistent with the image from which they
    came.

    Parameters
    ----------

    unary : a :class:`Layer` instance
        A layer providing values for the CRF unary potential. Generally the
        output of a per-pixel softmax nonlinearity. Shape must be:
        (batch_size, n_categories, H, W).

    ref : a :class:`Layer` instance
        A layer providing the reference images used for the CRF pairwise
        potentials. Typically this is just the input image layer. Shape
        must be: (batch_size, input_channels, H, W).

    sxy_bf : int
        Spatial standard deviation for the bilateral kernel of the CRF.

    sc_bf : int
        Color standard deviation for the bilateral kernel of the CRF. NOTE:
        Default values for this parameter assume the reference images are
        in the range [0,255] (or [-128,128] or a range similar magnitude).
        You should change this if your images do not satisfy this
        requirement.

    compat_bf : int
        Label compatibility weight for the bilateral kernel of the CRF. For
        now, assumes a Potts model w/a single parameter.

    sxy_spatial : int
        Spatial standard deviation for the 2D Gaussian kernel of the CRF.

    compat_spatial : int
        Label compatibility weight for the 2D Gaussian kernel of the CRF.
        For now, assumes a Potts model w/a single parameter.

    num_iter : int
        Number of inference iterations to perform.

    normalize_final_iter : bool
        Whether or not to normalize (apply softmax to) the output of the
        final inference iteration. If your model requires pre-softmax
        outputs, set this to False.

    trainable_kernels : bool
        By default, the kernel widths and compatibility weights (sxy_...,
        sc_..., compat_...) are fixed parameters. Set this to true to allow
        learning of these parameters (TODO: somewhat untested, maybe
        unstable).
    """
    def __init__(self, unary, ref, sxy_bf=70, sc_bf=10, compat_bf=6,
                 sxy_spatial=2, compat_spatial=2, num_iter=5,
                 normalize_final_iter=True, trainable_kernels=False,
                 name=None):

        super(CRFasRNNLayer, self).__init__(incomings=[unary, ref], name=name)

        self.sxy_bf = sxy_bf
        self.sc_bf = sc_bf
        self.compat_bf = compat_bf
        self.sxy_spatial = sxy_spatial
        self.compat_spatial = compat_spatial
        self.num_iter = num_iter
        self.normalize_final_iter = normalize_final_iter

        self.val_dim = ll.get_output_shape(unary)[1]
        # +2 for bilateral grid
        self.ref_dim = ll.get_output_shape(ref)[1] + 2

        kstd_bf = np.array([sxy_bf, sxy_bf, sc_bf, sc_bf, sc_bf], np.float32)
        self.kstd_bf = self.add_param(kstd_bf, (self.ref_dim,),
                                      name="kern_std",
                                      trainable=trainable_kernels,
                                      regularizable=False)

        gk = gkern(sxy_spatial, self.val_dim)
        self.W_spatial = self.add_param(gk, gk.shape, name="spatial_kernel",
                                        trainable=trainable_kernels,
                                        regularizable=False)

        if None in (self.val_dim, self.ref_dim):
            raise ValueError("CRF RNN requires known channel dimensions for \
all inputs.")

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        unary, ref = inputs

        N, _, H, W = ref.shape
        yx = tt.cast(tt.stack(tt.mgrid[0:H, 0:W]), "float32")
        grid = tt.alloc(yx[np.newaxis, :, :, :], N, 2, H, W)
        stacked = tt.concatenate([grid, ref], axis=1)

        def _bilateral(V, R):
            o = tt.ones((1, V.shape[1], V.shape[2]), "float32")
            norm = tt.sqrt(gaussian_filter(R, o, self.kstd_bf,
                                           self.ref_dim)) + 1e-8
            return gaussian_filter(R, V/norm, self.kstd_bf, self.ref_dim,
                                   self.val_dim) / norm

        def _step(prev_q, U, ref, normalize=True):
            qbf = _bilateral(prev_q, ref,)
            qsf = tt.nnet.conv2d(prev_q[np.newaxis, :, :, :],
                                 self.W_spatial, border_mode="half")[0]

            q_hat = -self.compat_bf * qbf + -self.compat_spatial * qsf
            q_hat = U - q_hat

            return softmax(q_hat, axis=0) if normalize else q_hat

        def _inference(unary_i, ref_i):
            U = tt.log(tt.clip(unary_i, 1e-5, 1))
            prev_q = softmax(U, axis=0)

            # This is faster than using scan.
            for i in range(self.num_iter):
                normalize = self.normalize_final_iter or i < self.num_iter-1
                prev_q = _step(prev_q, U, ref_i, normalize)
            return prev_q

        return theano.scan(fn=_inference, sequences=[unary, stacked],
                           outputs_info=None)[0]
