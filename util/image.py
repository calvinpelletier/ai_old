import numpy as np
from PIL import Image
import torch


def tensor2im(tensor, normalize=True, return_ndarray=False):
    # batch
    if tensor.dim() == 4:
        ret = []
        for b in range(tensor.size(0)):
            ret.append(tensor2im(tensor[b], normalize, return_ndarray))
        return ret

    # single
    im = tensor.detach().cpu().float().numpy()
    if normalize:
        im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        im = np.transpose(im, (1, 2, 0)) * 255.0
    im = np.clip(im, 0, 255)
    if im.shape[2] == 1:
        im = im[:, :, 0]

    if return_ndarray:
        return im.astype(np.uint8)
    else:
        return Image.fromarray(im.astype(np.uint8))
