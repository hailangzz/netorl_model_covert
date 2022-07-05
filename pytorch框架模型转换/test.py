
'''
import onnx

from onnxsim import simplify
import torch
from onnx_tf.backend import prepare

onnx_model = onnx.load("../ch_ppocr_mobile_v2.0_det_infer.onnx")
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"

output = prepare(model_simp)
output.export_graph("tf_model/")
print('Export tf_model!')
'''



import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
tflite_model = converter.convert()
open("ch_ppocr_mobile_v2.0_det_infer.tflite", "wb").write(tflite_model)
print('Export tf lite model!')
