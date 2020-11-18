import os

import cv2
import numpy as np
import torch as th
from tqdm import tqdm

from permutohedral.gfilt import gfilt

###############################################################################
# Perform a simple sanity check of the filtering / hashing components.  Creates
# a circle filled with a checkerboard pattern, blurs it using the circle's mask
# + XY coords as a reference image, and then optimizes an initial random image
# + initial kernel std. values through the filter to match the target image.
###############################################################################


def gaussian_filter(ref, val, kstd):
    return gfilt(ref / kstd[:, :, None, None], val)


sxy = 3
srgb = 0.25
h, w = 512, 512

if __name__ == "__main__":
    np.random.seed(0)
    th.manual_seed(0)

    yx = np.mgrid[:h, :w].astype(np.float32)
    yx = th.from_numpy(yx).cuda()
    kstd = th.FloatTensor([sxy, sxy, srgb]).cuda()

    tgt = np.zeros((512, 512, 3), np.uint8)
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (256, 256), 128, 255, -1)
    mask = mask > 0

    # Make a simple checkerboard texture.
    color_1 = [255, 128, 32]
    color_2 = [128, 255, 32]
    for i in range(8):
        for j in range(8):
            tgt[8 + i :: 16, 8 + j :: 16] = color_1
            tgt[i::16, j::16] = color_1
            tgt[8 + i :: 16, j::16] = color_2
            tgt[i::16, 8 + j :: 16] = color_2
    tgt[~mask] = 0
    tgt = th.from_numpy(tgt.transpose(2, 0, 1).copy()).cuda().float() / 255
    prefilt = tgt.clone()
    prefilt_np = (255 * prefilt).byte().data.cpu().numpy().transpose(1, 2, 0)

    # Create the filtered target image.
    stacked = th.cat([yx, th.from_numpy(mask[None]).cuda().float()], dim=0)
    tgt = gaussian_filter(stacked[None], tgt[None], kstd[None])[0]
    tgt_np = (255 * tgt).byte().data.cpu().numpy().transpose(1, 2, 0)

    # Create a random initial image that will be optimized.
    img_est = (0.5 * th.rand(3, h, w).cuda() + 0.25).requires_grad_(True)
    kstd_orig = kstd.clone()
    kstd[:] = 8
    kstd.requires_grad_(True)
    optim = th.optim.Adam([img_est, kstd], lr=1e-3)

    if not os.path.exists("sanity_imgs"):
        os.mkdir("sanity_imgs")

    for it in tqdm(range(8000)):
        filt = gaussian_filter(stacked[None], img_est[None], kstd[None])[0]
        diff = (filt - tgt) ** 2
        loss = diff.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # If left unconstrained, the image can take on negative values.
        with th.no_grad():
            img_est.clamp_(0, 1)

        if it == 4000:
            for pg in optim.param_groups:
                pg["lr"] /= 2

        if it % 25 == 0:
            with th.no_grad():
                filt = gaussian_filter(stacked[None], img_est[None], kstd[None])[0]
                diff = (filt - tgt) ** 2
                img = th.cat(
                    [
                        img_est.clamp(0, 1),
                        filt.clamp(0, 1),
                        (abs(diff) / diff.max().clamp(min=1e-2)).clamp(0, 1),
                    ],
                    dim=2,
                )
            img = (255 * img).byte().data.cpu().numpy().transpose(1, 2, 0)
            img = np.vstack([img, np.hstack([prefilt_np, tgt_np, np.zeros_like(tgt_np)])]).copy()

            color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Input (estimated)", (32, 64), font, 1, color, 1)
            cv2.putText(img, "Filtered (estimated)", (32 + 512, 64), font, 1, color, 1)
            cv2.putText(img, "Diff (filt_est - filt_gt)x100", (32 + 2 * 512, 64), font, 1, color, 1)
            cv2.putText(img, "Input (GT)", (32, 64 + 512), font, 1, color, 1)
            cv2.putText(img, "Filtered (GT)", (32 + 512, 64 + 512), font, 1, color, 1)
            cv2.imwrite(f"sanity_imgs/{it:06d}.png", img)
