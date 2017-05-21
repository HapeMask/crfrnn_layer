# CRF-as-RNN Layer for [Lasagne](https://github.com/lasagne/lasagne)

This repository contains a GPU-only implementation of the CRF-as-RNN method
[described here](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html). Please cite
their work if you use this in your own code. I am not affiliated with their
group, this is just a side-project.

The layer relies on two included [Theano](https://github.com/theano/theano)
ops: one to build the hashtable representing a [permutohedral
lattice](http://graphics.stanford.edu/papers/permutohedral/permutohedral.pdf)
and another to perform the high-dimensional Gaussian filtering required by
approximate CRF inference.

## Lasagne Layer
[![example](images/crf_layer_example.png)](images/crf_layer_example.png)

The [Lasagne](https://github.com/lasagne/lasagne) layer takes two inputs: a
probability map (typically the output of a softmax layer), and a reference
image (typically the image being segmented/densely-classified). Optional
additional parameters include:

* `sxy_bf`: spatial standard deviation for the bilateral filter.
* `sc_bf`: color standard deviation for the bilateral filter.
* `compat_bf`: label compatibility weight for the bilateral filter.
* `sxy_spatial`: spatial standard deviation for the 2D Gaussian filter.
* `compat_spatial`: label compatibility weight for the 2D Gaussian filter.

**Note**: the default color standard deviation assumes the input is a color
image in the range [0,255]. If you use whitened or otherwise-normalized images,
you should change this value.

Here is a simple example:

```python
import theano.tensor as tt
import lasagne.layers as ll
from lasagne.nonlinearities import rectify as relu

# Pixel-wise softmax is currently a WIP for Lasagne, this is a temporary helper.
def softmax(x, axis=1):
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

from crfrnn.layers import CRFasRNNLayer

n_categories = 150

inp = ll.InputLayer((None, 3, None, None))
conv1 = ll.Conv2DLayer(inp, num_filters=64, filter_size=3, pad="same",
                       nonlinearity=relu)
conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=3, pad="same",
                       nonlinearity=relu)
smax = ll.Conv2DLayer(conv2, num_filters=n_categories, filter_size=3,
                      pad="same", nonlinearity=softmax)

crf = CRFasRNNLayer(smax, inp)
```

## Theano Ops

The Theano ops used in the layer can also be used on their own for things like
bilateral filtering. [bilateral.py](bilateral.py) contains a sample
implementation.

`python bilateral.py input.png output.png 20 0.25`

<a
href="https://github.com/HapeMask/crfrnn_layer/raw/master/images/wimr_small.png"><img
src="https://github.com/HapeMask/crfrnn_layer/raw/master/images/wimr_small.png"
width=400 /></a> <a
href="https://github.com/HapeMask/crfrnn_layer/raw/master/images/filtered.png"><img
src="https://github.com/HapeMask/crfrnn_layer/raw/master/images/filtered.png"
width=400 /></a>
