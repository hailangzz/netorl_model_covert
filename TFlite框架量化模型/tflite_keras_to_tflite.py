import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(r'D:\PycharmProgram\netorl_model_covert\model_path\resnet50.h5')

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open(r"D:\PycharmProgram\netorl_model_covert\model_path\resnet50.tflite", "wb").write(tflite_model)

print(converter)
# # uint8 quant
# converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops = True
#
# converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}  # mean, std_dev
# converter.default_ranges_stats = (0, 255)
#
# tflite_uint8_model = converter.convert()
# open("model_path/model.tflite", "wb").write(tflite_uint8_model)