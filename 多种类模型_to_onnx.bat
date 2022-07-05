# 获取函数帮助···
python -m tf2onnx.convert -h
；

python -m tf2onnx.convert --opset 13 --saved-model {"D:/PycharmProgram/netorl_model_covert/model_path/resnet50.h5"} --output {"model_keras_to.onnx"}

# For checkpoint format:
python -m tf2onnx.convert --checkpoint ./model_path/net/my_net_ckpt.meta --output ./model_ckpt_to.onnx --inputs x_input:0 --outputs prob:0

# convert tflite models via command line, for example:
python -m tf2onnx.convert --tflite model_path/test.tflite --output model_tflit_to.onnx --opset=13 --dequantize

python -m tf2onnx.convert --tflite model_path/resnet50.tflite --output resnet50.onnx --opset=12 --dequantize

python -m tf2onnx.convert --tflite model_path/ch_ppocr_mobile_v2.0_det_infer.tflite --output ch_ppocr_mobile_v2.0_det_infer.onnx --opset=12 --dequantize


