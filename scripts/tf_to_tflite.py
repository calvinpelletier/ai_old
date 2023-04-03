#!/usr/bin/env python3
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    '/home/asiu/data/prod/client/v0_1/tf')
tflite_model = converter.convert()
with open('/home/asiu/data/prod/client/v0_1/model.tflite', 'wb') as f:
  f.write(tflite_model)
