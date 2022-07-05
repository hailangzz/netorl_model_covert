import tflite2onnx

tflite_path = r"../model_path/frozen_graph_model.tflite"
onnx_path = "../model_path/frozen_graph_model.onnx" #modelname.onnx


tflite2onnx.convert(tflite_path,onnx_path)