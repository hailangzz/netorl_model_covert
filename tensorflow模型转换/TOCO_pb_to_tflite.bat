toco  --graph_def_file ./model_path/on_device_vision_classify/saved_model.pb  --output_file ./frozen_graph_saved_model.tflite  --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE  --inference_type=QUANTIZED_UINT8  --input_shape=1, 321,321,3 --input_array=hub_input/images --output_array=transpose_1  --std_dev_value 127  --mean_value 127  --default_ranges_min 0  --default_ranges_max 255

toco --keras_model_file ./resnet50.h5 --output_file resnet50_h5_to.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE  --inference_type=QUANTIZED_UINT8 --input_shape=1,224,224,3 --input_array=input_1 --output_array=output


