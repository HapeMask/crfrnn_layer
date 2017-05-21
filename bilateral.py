import sys

import numpy as np
from scipy.misc import imread, imsave
import theano
import theano.tensor as tt

from crfrnn.gpugfilt import gaussian_filter

def usage():
    print("Usage: python bilateral.py input output sxy srgb")
    exit(1)

if len(sys.argv) != 5:
    usage()

try:
    sxy = float(sys.argv[3])
    srgb = float(sys.argv[4])
except:
    usage()

img = imread(sys.argv[1]).astype(np.float32)[..., :3] / 255.
img = img.transpose(2,0,1)
yx = np.mgrid[:img.shape[1], :img.shape[2]].astype(np.float32)
stacked = np.vstack([yx, img])

kstd = np.array([sxy, sxy, srgb, srgb, srgb], np.float32)

R = tt.tensor3("R")
I = tt.tensor3("I")

N = gaussian_filter(R, tt.ones_like(I[:1, :, :]), kstd, 5, 1)
F = gaussian_filter(R, I, kstd, 5, 3) / N

bilateral = theano.function([R,I], F)

out = np.asarray(bilateral(stacked, img))
imsave(sys.argv[2], out.transpose(1,2,0))
