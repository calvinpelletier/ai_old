#!/usr/bin/env python3
import os
import os.path
import pickle
import numpy as np
import torch
# import tensorflow as tf
# from dnnlib import tflib
# from global_directions.utils.visualizer import HtmlPageVisualizer
from external.e4e.models.stylegan2.model import Generator
from ai_old.util.stylespace import decoder


def load_data(dataset_name):
    path = f'/home/asiu/data/styleclip/npy/{dataset_name}/s.pkl'
    with open(path, 'rb') as fp:
        s_names, all_s = pickle.load(fp)
    stylespace = all_s

    pindexs = []
    mindexs = []
    for i in range(len(s_names)):
        name = s_names[i]
        if 'rgb' not in name:
            mindexs.append(i)
        else:
            pindexs.append(i)

    path = f'/home/asiu/data/styleclip/npy/{dataset_name}/s_mean_std.pkl'
    with open(path, 'rb') as fp:
        m, std = pickle.load(fp)

    return stylespace, s_names, mindexs, pindexs, m, std


def load_model():
    path = '/home/asiu/data/models/stylegan/stylegan2-ffhq-config-f.pt'
    ckpt = torch.load(path)
    model = Generator(1024, 512, 8).to('cuda')
    model.load_state_dict(ckpt['g_ema'], strict=False)
    model = model.eval()
    return model


def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False):
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])

    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    np.clip(images, 0, 255, out=images)
    images = images.astype('uint8')
    return images


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    if nhwc_to_nchw:
        images = np.rollaxis(images, 3, 1)
    return images / 255 * (drange[1] - drange[0]) + drange[0]


def _append_or_concat(arr, idx, val):
    if idx >= len(arr):
        arr.append(val)
    else:
        arr[idx] = torch.cat([arr[idx], val], dim=0)


class Manipulator():
    def __init__(self,dataset_name='ffhq'):
        self.dataset_name = dataset_name

        self.alpha = [0] # manipulation strength
        self.num_images = 10
        self.img_index = 0 # which image to start
        self.viz_size = 256
        self.manipulate_layers = None # which layer to manipulate, list

        self.stylespace, self.s_names, self.mindexs, self.pindexs, \
            self.code_mean, self.code_std = load_data(dataset_name)

        self.G = load_model()
        self.noise = [
            getattr(self.G.noises, 'noise_{}'.format(i)) \
            for i in range(self.G.num_layers)
        ]

        # self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.img_size = 1024

    def generate_img(self, ws, stylespace):
        num_images, step = stylespace[0].shape[:2]
        out = np.zeros(
            (num_images, step, self.img_size, self.img_size, 3),
            dtype='uint8',
        )
        with torch.no_grad():
            for i in range(num_images):
                for k in range(step):
                    img = decoder(
                        self.G,
                        [
                            torch.tensor(x[i, k]).cuda().unsqueeze(dim=0) \
                            for x in stylespace
                        ],
                        ws[i].cuda().unsqueeze(dim=0),
                        self.noise,
                    )
                    img = convert_images_to_uint8(
                        img.cpu().numpy(),
                        nchw_to_nhwc=True,
                    )
                    out[i, k, :, :, :] = img[0]
        return out

    def ms_code(self, dlatent_tmp, boundary_tmp):
        step = len(self.alpha)
        dlatent_tmp1 = [
            tmp.reshape((self.num_images, -1)) for tmp in dlatent_tmp
        ]
        dlatent_tmp2 = [
            np.tile(tmp[:, None], (1, step, 1)) for tmp in dlatent_tmp1
        ]

        l = np.array(self.alpha)
        l = l.reshape([
            step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)
        ])

        if type(self.manipulate_layers) == int:
            tmp = [self.manipulate_layers]
        elif type(self.manipulate_layers) == list:
            tmp = self.manipulate_layers
        elif self.manipulate_layers is None:
            tmp = np.arange(len(boundary_tmp))
        else:
            raise ValueError('manipulate_layers is wrong')

        for i in tmp:
            dlatent_tmp2[i] += l * boundary_tmp[i]

        codes = []
        for i in range(len(dlatent_tmp2)):
            tmp = list(dlatent_tmp[i].shape)
            tmp.insert(1, step)
            codes.append(dlatent_tmp2[i].reshape(tmp))
        return codes

    # def edit_one(self, bname, dlatent_tmp=None):
    #     if dlatent_tmp == None:
    #         dlatent_tmp = [
    #             tmp[self.img_index:(self.img_index+self.num_images)] \
    #             for tmp in self.stylespace
    #         ]
    #
    #     boundary_tmp=[]
    #     for i in range(len(self.boundary)):
    #         tmp=self.boundary[i]
    #         if len(tmp)<=bname:
    #             boundary_tmp.append([])
    #         else:
    #             boundary_tmp.append(tmp[bname])
    #
    #     codes=self.ms_code(dlatent_tmp,boundary_tmp)
    #
    #     out=self.generate_img(codes)
    #     return codes,out

    def edit_one_c(self, ws, cindex, dlatent_tmp=None):
        if dlatent_tmp == None:
            dlatent_tmp = [
                tmp[self.img_index:(self.img_index + self.num_images)] \
                for tmp in self.stylespace
            ]

        ws = ws[self.img_index:(self.img_index + self.num_images)]
        ws = ws.unsqueeze(dim=1).repeat(1, 18, 1)

        boundary_tmp = [[] for i in range(len(self.stylespace))]

        # only manipulate 1 layer and one channel
        assert len(self.manipulate_layers) == 1

        ml = self.manipulate_layers[0]
        tmp = dlatent_tmp[ml].shape[1] # ada
        tmp1 = np.zeros(tmp)
        tmp1[cindex] = self.code_std[ml][cindex] # 1
        boundary_tmp[ml] = tmp1

        codes = self.ms_code(dlatent_tmp, boundary_tmp)
        out = self.generate_img(ws, codes)
        return codes, out

    def w_to_style(self, ws):
        G = self.G
        print('ws', ws.shape)

        styles = []
        for w in ws:
            w = w.unsqueeze(0)

            idx = 0
            _append_or_concat(styles, idx, G.conv1.conv.modulation(w[:, 0]))
            idx += 1
            # _append_or_concat(styles, idx, G.to_rgb1.conv.modulation(w[:, 1]))
            # idx += 1

            i = 1
            for conv1, conv2, rgb in zip(G.convs[::2], G.convs[1::2], G.to_rgbs):
                _append_or_concat(styles, idx, conv1.conv.modulation(w[:, i]))
                idx += 1
                _append_or_concat(styles, idx, conv2.conv.modulation(w[:, i+1]))
                idx += 1
                # _append_or_concat(styles, idx, rgb.conv.modulation(w[:, i+2]))
                # idx += 1
                i += 2

        for i in range(len(styles)):
            styles[i] = styles[i].cpu().detach().numpy()
        return styles
