#!/usr/bin/env python3
from ai_old.dataset import DatasetBase
from ai_old.loss.clip import ClipLoss
import clip
import ai_old.dataset.filter_func as ff
import torch
import tqdm
import ai_old.constants as c


class DS(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('ffhq-128')

    def select_cols(self):
        return {
            'item_id': 'id',
            'face_image_1024': 'img',
        }

ds = DS().inference(batch_size=8, seed=0, rank=0, num_gpus=1)
clip_loss = ClipLoss(imsize=1024)
texts = [torch.cat([clip.tokenize(x)]).to('cuda') for x in c.CLIP_ATTRS]

with torch.no_grad():
    for batch in ds:
        img = batch['img'].to('cuda').to(torch.float32) / 127.5 - 1
        results = []
        for text in texts:
            results.append(clip_loss(img, text))
        results = torch.cat(results, dim=1)
        for i in range(8):
            print(batch['id'][i], results[i].cpu().numpy())
