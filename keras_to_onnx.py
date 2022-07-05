import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import onnxruntime
from tensorflow.keras.models import load_model

model = load_model("model_path/resnet50.h5")
# # 加载keras h5模型
# model = ResNet50(weights='imagenet')
# model.save(os.path.join("./tmp", model.name+'.h5'))

# h5模型转 onnx
import tf2onnx
import onnxruntime as rt
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
# output_names = [n.name for n in model_proto.graph.output]