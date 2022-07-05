# 在cmd命令中执行如下操作,主要针对infer,推理模型进行量化固定：
paddle2onnx  --model_dir D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --opset_version 11 --save_file ch_ppocr_mobile_v2.0_det_infer.onnx --input_shape_dict="{'x':[1, 3, 640, 640]}" --enable_onnx_checker True


paddle2onnx  --model_dir D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_PP-OCRv3_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --opset_version 11 --save_file ch_PP-OCRv3_rec_infer.onnx --input_shape_dict="{'x':[1, 3, 48, 144]}"




