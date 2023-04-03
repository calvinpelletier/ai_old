#!/usr/bin/env python3
import torch
import os
from ai_old.nn.models.prod import v0_1
from subprocess import check_call


FOLDER = '/home/asiu/data/prod/client/{}'


def export_v0_1():
    version = 'v0_1'
    prod_client = v0_1.get_prod_client()

    # convert to onnx
    onnx_path = os.path.join(FOLDER.format(version), 'model.onnx')
    torch.onnx.export(
        prod_client,
        (
            torch.zeros((1, 512, 4, 4)).to('cuda'),
            torch.zeros((1, 512, 4, 4)).to('cuda'),
            torch.zeros((1, 512)).to('cuda'),
            torch.zeros((1, 512)).to('cuda'),
            torch.zeros((1,)).to('cuda'),
        ),
        onnx_path,
        verbose=True,
        input_names=['base_enc', 'identity', 'base_latent', 'delta', 'mag'],
        output_names=['generated_img'],
        opset_version=11,
        # opset_version=13,
    )

    # convert onnx to tf
    tf_path = os.path.join(FOLDER.format(version), 'tf')
    check_call(' '.join([
        'onnx-tf',
        'convert',
        '-i', onnx_path,
        '-o', tf_path,
    ]), shell=True)

    # convert tf to tf.js
    tfjs_path = os.path.join(FOLDER.format(version), 'tfjs')
    check_call(' '.join([
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_node_names=\'generated_img\'',
        '--saved_model_tags=serve',
        tf_path,
        tfjs_path,
    ]), shell=True)


if __name__ == '__main__':
    export_v0_1()
