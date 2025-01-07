import random
import torch
from net.CNN_Net import CNN4,CNN9,CNN11,VGG8_8,VGG8_16,VGG8_32,ResNet18_8,ResNet18_16,ResNet18_32,DnCNN,CNN4,CNN7,CNN11
from net.Net import Activation_Net4
from torch.autograd import Variable
import numpy as np
import time
from torchsummary import summary
from thop import profile



#将数据放大k倍
def data_enlarge(data,k):
    for i in range(len(data)):
        temp = data[i]
        for j in range(len(temp)):
            data[i][j] = data[i][j]*k
    return data



if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)  # 全部输出


    input = np.ones((1,32,32),dtype=float)
    for num in range(1):
        for i in range(32):
            for j in range(32):
                input[num][i][j] = float(random.random())
    
    input = input.astype(np.float64)
    input = data_enlarge(input, 1000000)
    input = torch.tensor(input)
    input = torch.unsqueeze(input, dim=1)
    input.cuda()


    model = CNN9().double()
    device = torch.device("cuda:0")
    model.to(device)
    # model = VGG8_32().double()
    model.load_state_dict(torch.load(
        '../model/model16/vgg8/VGG8(100-0.005-batchsize=128).pt', map_location='cuda:0'))
    model.eval()
    
    start = time.perf_counter()
    model(input)
    end = time.perf_counter()
    print(end-start)

    model = CNN9()
    input = torch.randn(1, 1, 16, 16)
    #input = torch.randn(1, 1, 32, 32)
    Flops, params=profile(model, inputs=(input,))
    print('Flops: % .8fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .8fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值

    # model = VGG8_32().cuda()
    # summary(model, input_size=(1,32,32),batch_size=-1)

    # iterations = 300  # 重复计算的轮次
    #
    # model = CNN7().float()
    # device = torch.device("cuda:0")
    # model.to(device)
    #
    # random_input = torch.randn(1, 1, 16, 16).float()
    # print(random_input.dtype)
    # random_input = random_input.to(device)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #
    # # GPU预热
    # for _ in range(100):
    #     _ = model(random_input)
    #
    # # 测速
    # times = torch.zeros(iterations)  # 存储每轮iteration的时间
    # with torch.no_grad():
    #     for iter in range(iterations):
    #         starter.record()
    #         _ = model(random_input)
    #         ender.record()
    #         # 同步GPU时间
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)  # 计算时间
    #         times[iter] = curr_time
    #         #print(curr_time)
    #
    # mean_time = times.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))




