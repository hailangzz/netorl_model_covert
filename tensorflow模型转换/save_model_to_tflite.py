import tensorflow as tf
# savedModel 保存为 TFLite
saved_model_to_tflite_converter = tf.lite.TFLiteConverter.from_saved_model('../model_path/local-linearity_cifar10_1/saved_model/')
# 量化tflite 模型，可以在损失较小精度或不影响精度的情况下减小模型大小
saved_model_to_tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
saved_model_tflite = saved_model_to_tflite_converter.convert()

# with open('./tflite_models/saved_model_tflite', 'wb') as f:
#    f.write(saved_model_tflite)
# 保存量化的 tflite 模型
with open('../model_path/local-linearity_cifar10_1/saved_model/quantized_saved_model_tflite', 'wb') as f:
    f.write(saved_model_tflite)

#=================================================================================
'''
saved_model_cli show --dir model_path/ssd_mobilenet_v1_coco_2018_01_28/saved_model/ --tag_set serve  --signature_def serving_default


'''