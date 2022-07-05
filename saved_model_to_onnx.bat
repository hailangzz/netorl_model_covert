
# tensorflow 三种文件保存格式：分别为， .pb、.ckpt、 .save_model(服务器版格式)
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
# 写入序列化的 PB 文件
with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())

saver.save(sess,'./ckpt/my_model')

tf.compat.v1.saved_model.simple_save(sess,"./saved_model",inputs={"x_input": x},outputs={"output": prediction})
=======================================================================================================================

首先查看save_model、frozen graph 网络结构的输入输出信息：
1.A look at the model shows inputs and outptus of this model:
saved_model_cli show --dir model_path/ssd_mobilenet_v1_coco_2018_01_28/saved_model/ --tag_set serve  --signature_def serving_default
saved_model_cli show --dir model_path/centernet_resnet50v1_fpn_512x512_kpts_1/saved_model/ --tag_set serve  --signature_def serving_default

2.Lets start with the saved_model since this is in most cases the best and prefered format to convert from.
python -m tf2onnx.convert --opset 10 --saved-model model_path/ssd_mobilenet_v1_coco_2018_01_28/saved_model --output model_path/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_v1_coco_2018_01_28.onnx
python -m tf2onnx.convert --opset 12 --saved-model model_path/centernet_resnet50v1_fpn_512x512_kpts_1/saved_model --output model_path/centernet_resnet50v1_fpn_512x512_kpts_1/mnist.onnx


3.使用冻结图转换模型：
If we want use the the frozen graph we need to specify inputs and outputs.
python -m tf2onnx.convert --graphdef model_path/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --output model_path/ssd_mobilenet_v1_coco_2018_01_28/.frozen.onnx --opset 10  --inputs image_tensor:0 --outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0




