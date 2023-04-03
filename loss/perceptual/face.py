#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models.encode.resnet_backbone import Backbone
import ai_old.constants as c
import os

class FaceIdLoss(nn.Module):
    def __init__(self, imsize):
        super().__init__()
        self.input_imsize = imsize
        self.facenet = Backbone(
            input_size=112,
            num_layers=50,
            drop_ratio=0.6,
            mode='ir_se',
        )
        self.facenet.load_state_dict(torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'arcface/model_ir_se50.pth',
        )))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.crop_mult = imsize // 128

    def extract_feats(self, x):
        m = self.crop_mult
        x = x[:, :, int(17.5*m):int(111.5*m), (16*m):(110*m)]
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y, target, mean=True):
        bs = y.shape[0]
        assert y.shape == (bs, 3, self.input_imsize, self.input_imsize)
        assert target.shape == (bs, 3, self.input_imsize, self.input_imsize)
        embeddings1 = self.extract_feats(y)
        embeddings2 = self.extract_feats(target)
        embeddings2 = embeddings2.detach()

        # diff = 0.
        # for i in range(bs):
        #     cosine_similarity = embeddings1[i].dot(embeddings2[i])
        #     diff += 1. - cosine_similarity
        # loss = diff / bs

        loss = (1. - torch.bmm(
            embeddings1.view(bs, 1, -1),
            embeddings2.view(bs, -1, 1),
        ))

        if mean:
            return loss.mean()
        else:
            return loss


class SoloFaceIdLoss(nn.Module):
    def __init__(self, imsize, target_img):
        super().__init__()
        assert imsize >= 256

        self.facenet = Backbone(
            input_size=112,
            num_layers=50,
            drop_ratio=0.6,
            mode='ir_se',
        )
        self.facenet.load_state_dict(torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'arcface/model_ir_se50.pth',
        )))
        self.facenet = self.facenet.to('cuda')
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        target_enc = self.extract_feats(target_img)
        self.register_buffer('target_enc', target_enc.detach())

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, avg_batch=True):
        embeddings1 = self.extract_feats(x)
        embeddings2 = self.target_enc
        bs = x.shape[0]

        loss = (1. - torch.bmm(
            embeddings1.view(bs, 1, -1),
            embeddings2.view(bs, -1, 1),
        ))

        if avg_batch:
            return loss.mean()
        else:
            return loss.squeeze()
