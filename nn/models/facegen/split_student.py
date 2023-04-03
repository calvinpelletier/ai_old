#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from external.sg2.unit import SynthesisBlock
from ai_old.nn.models.facegen.fgbg import NonUpSynthesisBlock
from ai_old.util.pretrained import build_pretrained_sg2
from external.sg2 import persistence


@persistence.persistent_class
class Sg2SplitStudent(nn.Module):
    def __init__(self, cfg, freeze_first_half=True, init_first_half=True):
        super().__init__()
        self.freeze_first_half = freeze_first_half
        student_imsize = cfg.dataset.imsize

        self.g1 = Sg2StudentHalf(student_imsize, is_first_half=True)
        self.g2 = Sg2StudentHalf(student_imsize, is_first_half=False)

        if freeze_first_half:
            self.g1.requires_grad_(False)
            self.g1.eval()

        if init_first_half:
            pretrained = build_pretrained_sg2().synthesis
            for res in self.g1.block_resolutions:
                if res <= student_imsize:
                    getattr(self.g1, f'b{res}').load_state_dict(
                        getattr(pretrained, f'b{res}').state_dict(),
                    )

    def forward(self, ws, **block_kwargs):
        feat, img = self.g1(ws, None, None, **block_kwargs)
        _, out = self.g2(ws, feat, img, **block_kwargs)
        return out

    def prep_for_train_phase(self):
        if not self.freeze_first_half:
            self.g1.requires_grad_(True)
        self.g2.requires_grad_(True)


@persistence.persistent_class
class Sg2StudentHalf(nn.Module):
    def __init__(self,
        student_imsize,
        is_first_half,
        w_dim=512,
        teacher_imsize=1024,
        img_channels=3,
        channel_base=32768,
        channel_max=512,
        num_fp16_res=4,
        conv_clamp=256,
    ):
        assert teacher_imsize >= 4 and \
            teacher_imsize & (teacher_imsize - 1) == 0
        super().__init__()
        self.student_imsize = student_imsize
        self.is_first_half = is_first_half
        self.w_dim = w_dim
        self.teacher_imsize = teacher_imsize
        self.img_resolution_log2 = int(np.log2(teacher_imsize))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }
        fp16_resolution = max(
            2 ** (self.img_resolution_log2 + 1 - num_fp16_res),
            8,
        )

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.teacher_imsize)
            if res > student_imsize:
                block = NonUpSynthesisBlock(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                    resolution=student_imsize,
                    img_channels=img_channels,
                    is_last=is_last,
                    use_fp16=use_fp16,
                    conv_clamp=256,
                ) if not is_first_half else None
            else:
                block = SynthesisBlock(
                    in_channels,
                    out_channels,
                    w_dim=w_dim,
                    resolution=res,
                    img_channels=img_channels,
                    is_last=is_last,
                    use_fp16=use_fp16,
                    conv_clamp=256,
                ) if is_first_half else None
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            if block is not None:
                setattr(self, f'b{res}', block)

    def forward(self, ws, x, img, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        for res, cur_ws in zip(self.block_resolutions, block_ws):
            if (res <= self.student_imsize and self.is_first_half) or \
                    (res > self.student_imsize and not self.is_first_half):
                block = getattr(self, f'b{res}')
                x, img = block(x, img, cur_ws, **block_kwargs)
        return x, img
