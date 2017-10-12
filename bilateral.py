import sys

import numpy as np
import torch as th
from scipy.misc import imread, imsave

from crfrnn.functions import gfilt

def gaussian_filter(ref, val, kstd):
    return gfilt(ref / kstd[:, None, None], val)

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

img = th.from_numpy(img).cuda()
stacked = th.from_numpy(stacked).cuda()
kstd = th.FloatTensor([sxy, sxy, srgb, srgb, srgb]).cuda()

N = gaussian_filter(stacked, th.ones_like(img[:1]), kstd)
F = gaussian_filter(stacked, img, kstd) / N

F = F.data.cpu().numpy()
imsave(sys.argv[2], F.transpose(1,2,0))
