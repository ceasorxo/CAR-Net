import torch
import torch.nn
import onnx
from net.CNN_Net import CNN4_2,CNN9_2,CNN11_2

model = CNN4_2()
model = torch.load('model/model8/onnx/CNN_net4.pt')
#model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 1, 8, 8, requires_grad=True)

torch.onnx.export(model, x, 'model/model8/onnx/CNN_net4.onnx', input_names=input_names, output_names=output_names, verbose='True')