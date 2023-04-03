#!/usr/bin/env python3
import torch
import torch.nn as nn
from copy import deepcopy
from ai_old.util.factory import build_model_from_exp, build_dataset
from ai_old.util.etc import normalized_tensor_to_pil_img, create_img_row, \
    pil_to_tensor
from PIL import Image
from ai_old.util.face import align
from ai_old.loss.imsim import FaceImSimLoss
import os
from ai_old.nn.blocks.quant import QuantSimpleNoiseResUpConvBlock
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvToImg


DEVICE = 'cpu'


class QuantModel(nn.Module):
    def __init__(self,
        imsize=256,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
    ):
        super().__init__()
        self.n_blocks = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(self.n_blocks + 1)]
        nc = nc[::-1]

        for i in range(self.n_blocks):
            block = QuantSimpleNoiseResUpConvBlock(
                smallest_imsize * (2 ** (i + 1)),
                nc[i],
                nc[i+1],
            )
            setattr(self, f'b{i}', block)

        self.final = ConvToImg(nc[-1])

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        assert self.n_blocks == 6
        x = self.quant(x)
        x = self.b0(x, noise_mode='const')
        x = self.b1(x, noise_mode='const')
        x = self.b2(x, noise_mode='const')
        x = self.b3(x, noise_mode='const')
        x = self.b4(x, noise_mode='const')
        x = self.b5(x, noise_mode='const')
        x = self.dequant(x)
        return self.final(x)

    def fuse_model(self):
        for i in range(self.n_blocks):
            getattr(self, f'b{i}').fuse_model()


def evaluate(e, g, criterion, ds):
    loss_total = 0.
    cnt = 0
    with torch.no_grad():
        for batch in ds:
            img = batch['y'].to(DEVICE).to(torch.float32) / 127.5 - 1
            enc = e(img)
            out = g(enc)
            loss = criterion(out, img)
            cnt += 1
            loss_total += loss
    return loss_total / cnt


def print_size_of_model(model):
    torch.save(model.state_dict(), '/tmp/temp.p')
    print('Size (MB):', os.path.getsize('/tmp/temp.p') / 1e6)
    os.remove('/tmp/temp.p')


def run():
    test_img = align('/home/calvin/data/asi/sera/og/13.jpg', 1)[1]
    test_img = test_img.resize((256, 256), Image.LANCZOS)
    test_img = pil_to_tensor(test_img).to(DEVICE)

    ae, cfg = build_model_from_exp('rec/25/9', 'G_ema')
    e = deepcopy(ae.e)
    g = QuantModel()
    g.load_state_dict(ae.g.state_dict())
    # g = deepcopy(ae.g)
    del ae
    e = e.to(DEVICE).eval()
    g = g.to(DEVICE).eval()

    batch_size = 16
    dataset_core = build_dataset(cfg.dataset)
    val_set = dataset_core.get_val_set(
        batch_size,
        0, # seed
        0, # rank
        1, # num gpus
        verbose=False,
    )
    test_set = dataset_core.get_test_set(
        batch_size,
        0, # seed
        0, # rank
        1, # num gpus
        verbose=False,
    )

    criterion = FaceImSimLoss(256).to(DEVICE).eval()

    test_enc = e(test_img)

    print('BASELINE')
    print_size_of_model(g)
    loss = evaluate(e, g, criterion, test_set)
    print(f'loss: {loss}')
    im1 = g(test_enc)

    # print('FUSED')
    # g.fuse_model()
    # print_size_of_model(g)
    # loss = evaluate(e, g, criterion, test_set)
    # print(f'loss: {loss}')
    # im2 = g(test_enc)

    print('QUANTIZED')
    g.qconfig = torch.quantization.default_qconfig
    # g.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(g.qconfig)
    torch.quantization.prepare(g, inplace=True)
    evaluate(e, g, criterion, val_set)
    # g = g.to('cpu')
    torch.quantization.convert(g, inplace=True)
    print_size_of_model(g)
    loss = evaluate(e, g, criterion, test_set)
    print(f'loss: {loss}')
    im3 = g(test_enc)

    create_img_row([
        normalized_tensor_to_pil_img(x[0]) for x in [test_img, im1, im2, im3]
    ], 256).save('/home/calvin/data/asi/tmp/tmp.png')


if __name__ == '__main__':
    run()
