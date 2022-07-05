import tflite2onnx

tflite_path = "model_path/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.tflite"
onnx_path = "model_path/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.onnx" #modelname.onnx


tflite2onnx.convert(tflite_path,onnx_path)