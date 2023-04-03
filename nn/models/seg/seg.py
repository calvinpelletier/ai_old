#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.models.seg.deeplab import build_resnet101
import os
import ai_old.constants as c
from ai_old.util.params import requires_grad
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# from CelebAMask-HQ dataset
LABELS = [
    'bg', 'skin', 'nose', 'glasses', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
    'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'earring',
    'necklace', 'neck', 'cloth',
]
LABEL_TO_IDX = {x: i for i, x in enumerate(LABELS)}

# label colors for visualizing a face segmentation
COLORS = [
    [0, 0, 0],
    [204, 0, 0],
    [76, 153, 0],
    [204, 204, 0],
    [51, 51, 255],
    [204, 0, 204],
    [0, 255, 255],
    [255, 204, 204],
    [102, 51, 0],
    [255, 0, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [0, 0, 204],
    [255, 51, 153],
    [0, 204, 204],
    [0, 51, 0],
    [255, 153, 51],
    [0, 204, 0],
]
assert len(LABELS) == len(COLORS)

# min sum of hat pred across image to be considered a "has_hat" image
HAT_THRESHOLD = 500.
GLASSES_THRESHOLD = 50.

FHBC_GROUPS = [
    ['skin', 'nose', 'glasses', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
        'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'earring', 'neck'],
    ['hair', 'hat'],
    ['bg'],
    ['necklace', 'cloth'],
]


# for debugging purposes only
def colorize_seg(s, needs_argmax=True):
    if needs_argmax:
        s = torch.argmax(s, dim=1)
    s = s.cpu().numpy()
    b, h, w = s.shape
    imgs = np.zeros((b, h, w, 3))
    for i in range(b):
        for y in range(h):
            for x in range(w):
                imgs[i][y][x] = COLORS[s[i][y][x]]
    return [Image.fromarray(imgs[i].astype(np.uint8)) for i in range(b)]


def binary_seg_to_img(seg):
    return Image.fromarray(
        (seg * 256.).clamp(0, 255).to(torch.uint8).cpu().numpy(),
        'L',
    )

# TODO: add boolean for "multiple faces are in this photo"
"""
extracts various info from a face segmentation

input: seg tensor or seg tensor batch
output: dictionary or list of dictionaries
"""
class SegAnalyzer(Unit):
    def forward(self, s):
        dims = len(s.shape)
        if dims == 3:
            return self._get_info(s)
        elif dims == 4:
            return [self._get_info(s[i]) for i in range(s.shape[0])]
        else:
            raise ValueError(f'invalid input shape: {s.shape}')

    def _get_info(self, s):
        _, h, w = s.shape
        assert h == w == 128, 'seg analyzer was calibrated for 128x128 only'
        return {
            'mouth_size': self._extract_and_sum(s, 'mouth'),
            'has_glasses': self._extract_and_sum(s, 'glasses') > GLASSES_THRESHOLD,
            'has_hat': self._extract_and_sum(s, 'hat') > HAT_THRESHOLD,
        }

    def _extract_and_sum(self, s, label):
        return torch.sum(s[LABEL_TO_IDX[label], :, :].squeeze(dim=1))


"""
segments a face image

init:
    output imsize (if None, use resolution of input image)
    which seg label (if None, return all)
input:
    [b, 3, h, w] image tensor
    optional colorize flag for debugging
output:
    if colorize: [b, 3, h, w]
    elif label==none: [b, 19, h, w]
    else: [b, h, w]
"""
class Segmenter(Unit):
    def __init__(self, imsize=None, label=None):
        super().__init__()
        self.output_imsize = imsize
        self.model_imsize = 513 # lol
        self.label_idx = LABEL_TO_IDX[label] if label is not None else None

        # TODO: switch this over to our model loading system when it's ready
        # NOTE: the model loading system should pull from gcloud storage if not
        # already on disk
        print('Loading segmenter...')
        self.model = self._load_pretrained()
        print('Loaded.')

        # the pretrained model expects this image normalization (lol)
        self.segnorm = transforms.Normalize(
            [0.485, 0.456, 0.406], # rgb mean
            [0.229, 0.224, 0.225], # rgb std dev
        )

        # freeze model
        self.model.eval()
        requires_grad(self.model, False)

    def _load_pretrained(self):
        model = build_resnet101(
            pretrained=True,
            num_classes=19,
            num_groups=32,
            weight_std=True,
            beta=False,
        )
        cp = torch.load(os.path.join(
            c.PRETRAINED_MODELS,
            'deeplab/deeplab_model.pth',
        ))
        state_dict = {
            # i did not write this, idk wtf 7 is lol
            k[7:]: v for k, v in cp['state_dict'].items() if 'tracked' not in k
        }
        model.load_state_dict(state_dict)
        return model

    def forward(self,
        x,
        colorize=False,
        softmax=True,
        groups=None,
        output_imsize=None,
    ):
        if output_imsize is None:
            if self.output_imsize is None:
                _, _, h, w = x.shape
                assert h == w
                output_imsize = w
            else:
                output_imsize = self.output_imsize

        # resize to model imsize
        x = F.interpolate(
            x,
            size=(self.model_imsize, self.model_imsize),
            mode='bilinear',
            align_corners=False,
        )

        # segment
        s = self.model(self.segnorm(x))

        # resize to output size
        s = F.interpolate(
            s,
            size=(output_imsize, output_imsize),
            mode='bilinear',
            align_corners=False,
        )

        # for debugging
        if colorize:
            return colorize_seg(s)

        if softmax:
            s = F.softmax(s, dim=1)

        if self.label_idx is not None:
            assert groups is None
            return s[:, self.label_idx, :, :].squeeze(dim=1)

        if groups is not None:
            assert self.label_idx is None
            ret = torch.zeros(
                s.shape[0],
                len(groups),
                output_imsize,
                output_imsize,
                dtype=s.dtype,
                device=s.device,
            )
            for i, group in enumerate(groups):
                for label in group:
                    ret[:, i, :, :] += s[:, LABEL_TO_IDX[label], :, :]
            return ret

        return s
