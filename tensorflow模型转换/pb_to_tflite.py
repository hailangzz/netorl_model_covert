# -*- coding:utf-8 -*-
import tensorflow as tf

in_path = "../model_path/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb"
out_path = "../model_path/mobilenet_v1_1.0_224/mobilenet_v1.tflite"
# out_path = "./model/quantize_frozen_graph.tflite"

# 模型输入节点
input_tensor_name = ["input"]
input_tensor_shape = {"input":[1, 224,224,3]}
# 模型输出节点
classes_tensor_name = ["MobilenetV1/Predictions/Reshape_1"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
                                            input_tensor_name, classes_tensor_name,
                                            input_shapes = input_tensor_shape)
converter.post_training_quantize = True
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)

