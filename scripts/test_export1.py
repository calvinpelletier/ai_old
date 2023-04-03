#!/usr/bin/env python3
import torch
from ai_old.util.pretrained import build_pretrained_sg2
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row
import numpy as np
from ai_old.util.factory import build_model_from_exp
from ai_old.util.etc import resize_imgs
from ai_old.nn.models.lerp.onnx import OnnxLevelsDynamicLerper
from ai_old.nn.models.facegen.student import Sg2Student


IMSIZE = 256


class TestProdClientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        server_model, cfg = build_model_from_exp('lerp/5/5', 'G')
        self.f = OnnxLevelsDynamicLerper(
            levels=cfg.model.levels,
            final_activation=cfg.model.final_activation,
            mult=cfg.model.mult,
            lr_mul=cfg.model.lr_mul,
        )
        self.f.load_state_dict(server_model.f.state_dict())

        client_model, cfg = build_model_from_exp('distill/0/3', 'G_ema')
        self.g = Sg2Student(cfg, nc_base=cfg.model.G.nc_base)
        self.g.load_state_dict(client_model.state_dict())

    def forward(self, w_plus, gender, magnitude):
        new_w_plus = self.f(w_plus, gender, magnitude=magnitude)
        return self.g(new_w_plus)


class TestProdServerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = build_model_from_exp('lerp/5/5', 'G', return_cfg=False).f
        self.g = build_pretrained_sg2().synthesis

    def forward(self, w_plus, gender, magnitude):
        new_w_plus = self.f(w_plus, gender, magnitude=magnitude)
        img = self.g(new_w_plus, noise_mode='const')
        return resize_imgs(img, IMSIZE)


ws = torch.from_numpy(
    np.load('/home/asiu/data/sera/w/12.npy'),
).unsqueeze(0).to('cuda').to(torch.float32)

# server_G = build_pretrained_sg2().synthesis
server_G = TestProdServerModel()
server_G = server_G.to('cuda').eval()

# client_G = build_model_from_exp('distill/0/debug', 'G_ema', return_cfg=False)
client_G = TestProdClientModel()
client_G = client_G.to('cuda').eval()

gender = torch.ones(1, device='cuda')
assert gender.shape == (1,)

with torch.no_grad():
    server_imgs = server_G(ws, gender, 0.)
    client_imgs = client_G(ws, gender, 0.)
    server_swap_imgs = server_G(ws, gender, 1.)
    client_swap_imgs = client_G(ws, gender, 1.)

create_img_row([
    normalized_tensor_to_pil_img(server_imgs[0]),
    normalized_tensor_to_pil_img(client_imgs[0]),
    normalized_tensor_to_pil_img(server_swap_imgs[0]),
    normalized_tensor_to_pil_img(client_swap_imgs[0]),
], IMSIZE).save('/home/asiu/data/tmp/convert/comparison.png')

torch.onnx.export(
    client_G,
    (ws, gender, torch.tensor(1.).to('cuda')),
    '/home/asiu/data/tmp/convert/g.onnx',
    verbose=True,
    input_names=['w_plus', 'gender', 'magnitude'],
    output_names=['generated_img'],
    opset_version=11,
    # opset_version=13,
)
