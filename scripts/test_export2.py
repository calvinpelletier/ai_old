#!/usr/bin/env python3
from subprocess import call

onnx_path = '/home/asiu/data/tmp/convert/g.onnx'
tf_path = '/home/asiu/data/tmp/convert/tf'
tfjs_path = '/home/asiu/data/tmp/convert/tfjs'

call(' '.join([
    'onnx-tf',
    'convert',
    '-i', onnx_path,
    '-o', tf_path,
]), shell=True)

call(' '.join([
    'tensorflowjs_converter',
    '--input_format=tf_saved_model',
    '--output_node_names=\'generated_img\'',
    '--saved_model_tags=serve',
    tf_path,
    tfjs_path,
]), shell=True)
