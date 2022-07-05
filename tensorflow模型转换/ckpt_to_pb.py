import tensorflow as tf


def saveQuantCkpt2Pb(in_ckpt_dir,output_node_names):

    graph = tf.compat.v1.get_default_graph()  # 获取默认图
    tf.contrib.quantize.create_eval_graph(graph)  # 在默认图基础上创建一个推理图,使其包含量化结点

    sess = tf.compat.v1.Session(graph=graph)  # 创建一个会话，设置会话的图为graph

    # 创建一个模型保存和加载对象
    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.latest_checkpoint(in_ckpt_dir)  # 查找in_ckpt_dir目录下最新保存的checkpoint文件的文件名
    print("ckpt:================", ckpt)
    saver.restore(sess, ckpt)   # 将量化训练的ckpt模型参数加载到当前默认会话

    # 同时将网络模型结构与参数用二进制格式的pb文件保存
    # sess.graph.as_graph_def()：导出当前计算图的GraphDef部分，GraphDef保存了从输入层到输出层的计算过程
    converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)  # #保存指定的节点，并将节点值保存为常数
    # tf.gfile.GFile(filename, mode)：获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
    # tf.gfile.Open()是该接口的同名，可任意使用其中一个
    # with tf.io.gfile.GFile(pb_graph_dir, "wb") as f:   # 保存方法一
    #     f.write(converted_graph_def.SerializeToString())  # SerializeToString()序列化

    tf.io.write_graph(  # 保存方法二
        converted_graph_def,
        "./pb",
        "freeze_eval_graph.pb",
        as_text=False)


in_ckpt_dir=r'./ckpt'
output_node_names=['output']
saveQuantCkpt2Pb(in_ckpt_dir,output_node_names)