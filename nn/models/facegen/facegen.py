#!/usr/bin/env python3
import os
import ai_old.constants as c
import ai_old.nn.models.facegen.stylegan as sg
import torch
import torch.nn.functional as F
from ai_old.nn.models.seg import Segmenter
from ai_old.nn.models import Unit
from ai_old.util.params import requires_grad


"""
combo unit that generates faces and segments them
(to prevent unnecessary resizing)

init: output image size
input: disentangled latent vector of size (batch_size, 512)
outputs:
    img (b, 3, h, w)
    seg (b, 19, h, w)
"""
class FaceGenAndSeg(Unit):
    def __init__(self, imsize=128):
        super().__init__()
        self.imsize = imsize
        self.g = FaceGenerator(imsize=513)  # 513 is Segmenter's ideal imsize
        self.seg = Segmenter(imsize=imsize)

    def forward(self, z):
        y = self.g(z, is_entangled=False)
        s = self.seg(y)
        y = F.interpolate(
            y,
            size=(self.imsize, self.imsize),
            mode='bilinear',
            align_corners=False,
        )
        return y, s


"""
generate faces using stylegan2

input:
    latent vector of dims (batch_size, 512)
    bool indicating whether the latent vector is entangled (i.e. random noise)
output: img if input was disentangled otherwise (img, disentangled z)
"""
class FaceGenerator(Unit):
    def __init__(self, imsize=128, truncation=0.7):
        super().__init__()
        self.imsize = imsize

        # NOTE: 0.7 default trunc was chosen based on fig 4b in
        # https://arxiv.org/pdf/1904.06991.pdf
        self.truncation = truncation

        # TODO: switch this over to our model loading system when it's ready
        # NOTE: the model loading system should pull from gcloud storage if not
        # already on disk
        print('Loading face generator...')
        self.model, self.z_avg = self._load_pretrained()
        self.z_avg = self.z_avg.to('cuda')
        print('Loaded.')

        # freeze model
        self.model.eval()
        requires_grad(self.model, False)

    def forward(self, z, is_entangled=False):
        if is_entangled:
            # disentangle and truncate z, then generate
            y, z = self.model(
                z.unsqueeze(dim=0),  # model expects list of styles (for mixing)
                truncation=self.truncation,
                truncation_latent=self.z_avg,
                return_latents=True,
            )
        else:
            # only generate (z has already been disentangled)
            y, _ = self.model(z.unsqueeze(dim=0), input_is_latent=True)

        # resize image
        if self.imsize != 1024:
            y = F.interpolate(
                y,
                size=(self.imsize, self.imsize),
                mode='bilinear',
                align_corners=False,
            )

        # return disentangled z if input was entangled
        if is_entangled:
            assert (z[:, 0, :] == z[:, 1, :]).all()
            return y, z[:, 0, :].squeeze()

        return y

    def _load_pretrained(self):
        model = sg.Generator(1024, 512, 8)
        ckpt = torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'stylegan/stylegan2-ffhq-config-f.pt',
        ))
        model.load_state_dict(ckpt['g_ema'])
        return model, ckpt['latent_avg']
