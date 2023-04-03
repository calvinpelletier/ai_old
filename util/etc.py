import torch
import math
import random
import numpy as np
from time import time
from functools import wraps
from typing import Any, List, Tuple, Union
import importlib
import torch.nn.functional as F
from PIL import Image


def pil_to_tensor(img):
    img_np = np.asarray(img).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.copy(img_np))
    img_tensor = img_tensor.to('cuda').to(torch.float32) / 127.5 - 1
    return img_tensor.unsqueeze(0)


def nearest_lower_power_of_2(x):
    return 2**(int(math.log2(x)))


def resize_imgs(imgs, imsize):
    return F.interpolate(
        imgs,
        size=(imsize, imsize),
        mode='bilinear',
        align_corners=True,
    )


def tensor_to_pil(tensor):
    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor = (tensor * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(
        np.transpose(tensor, (1, 2, 0)),
        'RGB',
    )


def create_img_row(imgs, imsize, mode='RGB'):
    canvas = Image.new(
        mode,
        (imsize * len(imgs), imsize),
        'black',
    )
    for i, img in enumerate(imgs):
        canvas.paste(img, (imsize * i, 0))
    return canvas


def create_img_grid(imgs, imsize):
    canvas = Image.new(
        'RGB',
        (imsize * len(imgs[0]), imsize * len(imgs)),
        'black',
    )
    for y, row in enumerate(imgs):
        for x, img in enumerate(row):
            if img is not None:
                canvas.paste(img, (imsize * x, imsize * y))
    return canvas


def check_model_equality(a, b):
    assert len(list(a.parameters())) == len(list(b.parameters()))
    for p1, p2 in zip(a.parameters(), b.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def print_(rank, msg):
    if rank == 0:
        print(msg)


def timed(fn):
    @wraps(fn)
    def _impl(self, *args, **kwargs):
        start = time()
        ret = fn(self, *args, **kwargs)
        self.times[fn.__name__] = time() - start
        if not self.done_calculating_timed_fn_order:
            if fn.__name__ not in self.timed_fn_order:
                self.timed_fn_order.append(fn.__name__)
            else:
                self.done_calculating_timed_fn_order = True
        return ret
    return _impl


def make_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_acc(y_pred, y_target):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_sum = (y_pred_tag == y_target).sum().float()
    return 100. * correct_sum / y_target.shape[0]


def multiclass_acc(y, target):
    pred = torch.argmax(y, dim=1)
    correct = (pred == target).sum().float()
    return 100. * correct / target.shape[0]


def is_power_of_two(x):
    return isinstance(x, int) and (x > 0) and (x & (x-1) == 0)


def int_log2(x):
    assert is_power_of_two(x)
    return int(math.log2(x))


def log2_diff(a, b):
    return abs(int_log2(a) - int_log2(b))
