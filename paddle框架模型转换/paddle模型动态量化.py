import paddle
import paddleslim

# 开启静态图模式
paddle.enable_static()

# 模型的路径和文件名称
model_dir = r"D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer"
model_filename = 'inference.pdmodel'
params_filename = 'inference.pdiparams'
model_dir_quant_dynamic = r"D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer"

import os
import paddle
import paddleslim
import numpy as np
import paddle.vision.transforms as T

from PIL import Image

# 开启静态图模式
paddle.enable_static()

# 模型的路径和文件名称
model_dir = r"D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer"
model_filename = 'inference.pdmodel'
params_filename = 'inference.pdiparams'
model_dir_quant_static = r"D:\PycharmProgram\PaddleOCR-release-2.5\model\ch_ppocr_mobile_v2.0_det_infer"

# 数据预处理
'''
    缩放 -> 中心裁切 -> 类型转换 -> 转置 -> 归一化 -> 添加维度
'''
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transforms = T.Compose(
    [
        T.Resize(256, interpolation="bilinear"),
        T.CenterCrop(224),
        lambda x: np.asarray(x, dtype='float32').transpose(2, 0, 1) / 255.0,
        T.Normalize(mean, std),
        lambda x: x[None, ...]
    ]
)

# 校准数据读取
'''
    读取图像 -> 预处理 -> 组成数据字典
'''
img_dir = r'E:\data\flower_photos\roses'
img_num = 641
datas = iter([
    val_transforms(
        Image.open(os.path.join(img_dir, img)).convert('RGB')
    ) for img in os.listdir(img_dir)[:img_num]
])

# 静态量化
paddleslim.quant.quant_post_static(
    executor=paddle.static.Executor(),  # Paddle 静态图执行器
    model_dir=model_dir,  # 输入模型路径
    model_filename=model_filename,  # 输入模型计算图文件名称
    params_filename=params_filename,  # 输入模型参数文件名称
    quantize_model_path=model_dir_quant_static,  # 输出模型路径
    save_model_filename=model_filename,  # 输出模型计算图文件名称
    save_params_filename=params_filename,  # 输出模型参数文件名称
    batch_generator=None,  # 数据批次生成器，需传入一个可调用对象，返回一个 Generator
    sample_generator=lambda: datas,  # 数据采样生成器，需传入一个可调用对象，返回一个 Generator

    batch_size=32,  # 数据批次大小
    batch_nums=1,  # 数据批次数量，默认为使用全部数据
    weight_bits=8,  # 参数量化比特数 8/16 对应 INT8/16 类型
    activation_bits=8,  # 激活值量化比特数 8/16 对应 INT8/16 类型
    weight_quantize_type='channel_wise_abs_max',  # 参数量化方法，目前支持 'range_abs_max', 'moving_average_abs_max' 和 'abs_max'
    activation_quantize_type='range_abs_max',  # 激活值量化方法，目前支持 'range_abs_max', 'moving_average_abs_max' 和 'abs_max'
    algo='KL',  # 校准方法，目前支持 'KL', 'hist', 'mse', 'avg', 'abs_max'
)