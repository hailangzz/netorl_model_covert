import onnx
from onnx_tf.backend import prepare
import os

def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


if __name__ == "__main__":
    os.makedirs("tensorflow", exist_ok=True)
    onnx_input_path = 'D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer/ch_ppocr_mobile_v2.0_det_infer.onnx'
    pb_output_path = './tensorflow/model.pb'

    onnx2pb(onnx_input_path, pb_output_path)
