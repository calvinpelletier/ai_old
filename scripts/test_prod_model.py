#!/usr/bin/env python3
from ai_old.nn.models.prod import get_prod_model
import ai_old.dataset.filter_func as ff
import torch
from ai_old.dataset import DatasetBase
from torchvision.utils import save_image
from tqdm import tqdm
import ai_old.dataset.metadata_column_processor as mcp


class ffhqds(DatasetBase):
    def filter_func(self):
        return ff.for_dataset('ffhq-128')

    def select_cols(self):
        return {
            'item_id': 'id',
            'face_image': 'face',
        }

    def get_column_processor_overrides(self):
        return {
            'face_image': mcp.CP(
                inplace_method=mcp.read_image_without_norm,
            )
        }

dataset = ffhqds().inference(batch_size=8)
model = get_prod_model(exp='blend-ult/1/0')

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataset)):
        face = batch['face'].to('cuda')
        out = model(face, True, debug=True)
        for id, og, rec, swap in zip(batch['id'], face, out['rec'], out['swap']):
            save_image(
                [og, rec, swap],
                f'/home/asiu/data/tmp/prodtest/{id}.png',
                normalize=True,
                range=(-1, 1),
            )
