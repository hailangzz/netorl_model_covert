import torch
import torchvision
import torch.nn as nn

# 加载模型（根据自己模型修改结构和参数）
model_path ='D:\PycharmProgram\pytorch-model\FeatureRecognition-master\References\RFB_Net_vgg_1x3x300x300.pt'#
checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)

# 给模型的forward()方法一个示例输入
example = torch.rand(3, 3, 300, 300)
traced_script_module = torch.jit.trace(checkpoint, example)
# 保存模型
traced_script_module.save("./script_model.pt")

print("Finished Transformation")