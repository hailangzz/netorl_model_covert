import torch
device = 'cpu'

model_path ='./RFB_Net_vgg_1x3x300x300.pt'#
model = torch.load(model_path, map_location=lambda storage, location: storage)

print(model)
dummy_input = torch.randn(1,3,300,300)
onnx_path ='./RFB_Net_vgg_1x3x300x300.onnx'
torch.onnx.export(model, dummy_input, onnx_path, input_names=['x'], output_names=['o1','o2'],opset_version=12)
print('convert retinaface to onnx finish')
#注：其只在docker环境下可以完成转换
######################################################################################################################
# network.load_state_dict(checkpoint['model_state_dict'])
# network.eval().to(device)
#
# dummy_input = torch.zeros(64, 6, 7).to(device)
#
# input_names = ["input"]
# output_names = ["output"]
#
# torch.onnx.export(network, dummy_input, "resnet.onnx", verbose=True, input_names=input_names,
#                   output_names=output_names)