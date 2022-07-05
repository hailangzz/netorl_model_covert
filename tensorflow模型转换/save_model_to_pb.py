# import tensorflow as tf
# from tensorflow.python.framework import graph_util
#
# with tf.Session(graph=tf.Graph()) as sess:
#   tf.saved_model.loader.load(sess, ["serve"], "../model_path/mnist/saved_model/")
#   graph = tf.get_default_graph()
#
#   print(graph)
  # with tf.gfile.FastGFile('./model-mnist.pb', mode='wb') as f:
  #     f.write(graph.SerializeToString())




from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
sess = tf.Session()
tf.saved_model.loader.load(sess, ["serve"], r"../model_path/centernet_resnet50v1_fpn_512x512_kpts_1/saved_model/")
graph = tf.get_default_graph()
# 获取输入输出的tensor_name，而这个是在定义自己的operation的时候指定的

output_graph_def = convert_variables_to_constants(sess, sess.graph_def,
                                                    output_node_names=['StatefulPartitionedCall'])
with tf.gfile.FastGFile('./efficientdet_d0_coco17_tpu.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())

