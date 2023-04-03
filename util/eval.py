#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ai_old.util.image import tensor2im
from ai_old.util.etc import binary_acc
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import math
from functools import wraps


def possibly_redundant_eval_metric_calc(fn):
    @wraps(fn)
    def _impl(self, data, out, metric):
        # sanity
        if metric not in self.eval_metrics:
            raise Exception('tried to calc metric that was not configured: ' + \
                metric)

        # copy from val if possible
        if metric in self.val_metrics:
            self.eval_metrics[metric] = self.val_metrics[metric]
            return

        # calculate metric
        fn(self, data, out, metric)
    return _impl


def image_gen_eval(results_path, step, dataset, g):
    n_samples = 10
    ephemeral_path = os.path.join(results_path, 'generated', 'tmp')
    if not os.path.exists(ephemeral_path):
        os.makedirs(ephemeral_path)
    samples_path = os.path.join(results_path, 'generated', 'samples')
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    # generate images
    count = 0
    g.eval()
    with torch.no_grad():
        for data in dataset:
            out = tensor2im(g(data['a'].cuda()))
            for i in range(len(out)):
                # save all val images in ephemeral for metrics calc
                out[i].save(os.path.join(ephemeral_path, data['filename'][i]))
                if count < n_samples:
                    # save a few samples in permanent storage for inspection
                    out[i].save(os.path.join(samples_path, '{}_{}'.format(
                        step, data['filename'][i])))
                count += 1
    g.train()

    # calc fid
    # TODO


def ssim(x, y):
    ret = 0.
    x = tensor2im(x, return_ndarray=True)
    y = tensor2im(y, return_ndarray=True)
    for i in range(len(x)):
        im1 = img_as_float(rgb2gray(x[i]))
        im2 = img_as_float(rgb2gray(y[i]))
        score, _diff = structural_similarity(im1, im2, full=True)
        ret += score
    return ret / len(x)
