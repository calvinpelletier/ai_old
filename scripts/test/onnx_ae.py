#!/usr/bin/env python3
import torch
from ai_old.util.factory import build_model_from_exp, build_model
from ai_old.util.factory import build_dataset
from tqdm import tqdm
from copy import deepcopy
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_grid


ae1, cfg = build_model_from_exp('rec/25/8', 'G_ema')
ae1 = ae1.eval().requires_grad_(False).to('cuda')

setattr(cfg.model.G, 'onnx', True)
ae2 = build_model(cfg, cfg.model.G).eval().requires_grad_(False).to('cuda')
ae2.load_state_dict(ae1.state_dict(), strict=False)

batch_size = 8
dataset_core = build_dataset(cfg.dataset)
val_set = dataset_core.get_val_set(
    batch_size,
    0, # seed
    0, # rank
    1, # num gpus
    verbose=False,
)

with torch.no_grad():
    for batch in val_set:
        img = batch['y'].to('cuda').to(torch.float32) / 127.5 - 1
        enc1 = ae1.e(img)
        enc2 = ae2.e(img)
        rec1 = ae1.g(enc1, noise_mode='const')
        rec2 = ae2.g(enc2, noise_mode='const')
        print(rec1 - rec2)
        create_img_grid([
            [normalized_tensor_to_pil_img(x) for x in rec1],
            [normalized_tensor_to_pil_img(x) for x in rec2],
        ], 256).save('/home/asiu/data/test/onnx_conv.png')
        break

torch.onnx.export(
    ae2.g,
    (torch.zeros((1, 512, 4, 4)).to('cuda'),),
    '/home/asiu/data/test/onnx_conv.onnx',
    verbose=True,
    input_names=['enc'],
    output_names=['generated_img'],
    opset_version=11,
    # opset_version=13,
)
