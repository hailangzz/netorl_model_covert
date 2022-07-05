# import onnx
# from onnxruntime.quantization import quantize_dynamic, QuantType
#
# model_fp32 = '../RFB_Net_vgg_1x3x300x300.onnx'
# model_quant = '../RFB_Net_vgg_1x3x300x300_quant.onnx'
# quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8,)

import onnx
# from onnxruntime.quantization import quantize_qat, QuantType
#
# model_fp32 = '../RFB_Net_vgg_1x3x300x300.onnx'
# model_quant = '../RFB_Net_vgg_1x3x300x300_quant.onnx'
# quantized_model = quantize_qat(model_fp32, model_quant)


import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

# Update the input name and path for your ONNX model
input_onnx_model = '../ch_ppocr_mobile_v2.0_det_infer.onnx'
# Change this path to the output name and path for your float16 ONNX model
output_onnx_model = '../ch_ppocr_mobile_v2.0_det_infer-f16.onnx'
# Load your model
onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# Convert tensor float type from your input ONNX model to tensor float16
onnx_model = convert_float_to_float16(onnx_model)
# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)




