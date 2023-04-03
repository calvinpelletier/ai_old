#!/usr/bin/env python3
from ai_old.util.factory import convert_importpath


def get_prod_model(base='ai_old.nn.models.prod.v0.ProdV0', exp='blend-ult/0/0'):
    model_cls = convert_importpath(base)
    return model_cls(exp).eval().to('cuda')
