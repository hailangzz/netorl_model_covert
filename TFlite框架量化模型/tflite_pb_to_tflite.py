import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph_def_file = r"D:\PycharmProgram\netorl_model_covert\model_path\model.pb"

input_names = ["x"]
output_names = ["output"]
input_tensor = {input_names[0]: [1, 784]}

# uint8 quant
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)

tflite_uint8_model = converter.convert()
open(r"D:\PycharmProgram\netorl_model_covert\model_pathmodel.tflite", "wb").write(tflite_uint8_model)