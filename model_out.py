import random
import torch
from torch import nn,optim
from CNN_Net import CNN11,ConvBNRelu,BaseInception
from torch.autograd import Variable
import numpy as np
import xlwt
np.set_printoptions(threshold=np.inf)  #设置阈值为无限


def data_loader(file_route):
    file = open(file_route)

    data_load_in = []
    byt = file.readlines()
    for i in range(len(byt)):
        temp = byt[i].strip()
        temp = temp.split(' ')
        temp = np.array(temp)
        data = [float(i) for i in temp]
        data_load_in.append(data)
    return data_load_in


#将数据放大k倍
def data_enlarge(data,k):
    for i in range(len(data)):
        temp = data[i]
        for j in range(len(temp)):
            data[i][j] = data[i][j]*k
    return data


if __name__ == '__main__':
    file_route_in = "data/sim32p/48000/cmeans32x32.txt"

    #训练数据导入
    data_in_o = data_loader(file_route_in)
    data_in_o = np.array(data_in_o)
    data_in = np.zeros((48000, 32, 32), dtype=np.float64)
    for i in range(len(data_in_o)):
        data_in[i] = data_in_o[i].reshape(32, 32)
    data_in = data_in.astype(np.float64)
    print(data_in.shape)
    data_in = data_enlarge(data_in, 1000000)

    data_in = torch.tensor(data_in)
    data_in = torch.unsqueeze(data_in, dim=1)

    data_out = []

    # 导入模型
    model = CNN11().double()
    model.load_state_dict(torch.load('model/model32/CNN/48000/CNN_net11_Drop0.3(32x32x1-(3x1+1x3+1x5+5x1)x16-32x32x128-250-0.0001-batchsize=64).pt'))

    #data_in = Variable(data_in)
    for i in range(4800):
        out = model(data_in[i*10:i*10+10])
        out = out/1000000
        out = torch.squeeze(out, dim=1)
        temp = out.detach().numpy()
        #print(temp.shape)
        data_out.append(temp)

    data_out = np.array(data_out)
    print(data_out.shape)
    out = np.zeros((48000, 1024), dtype=np.float64)
    for i in range(4800):
        for j in range(10):
            for k in range(32):
                for m in range(32):
                    out[i*10+j,k*32+m] = data_out[i][j][k][m]

    f = open("r_vcmat32x32.txt", "w")
    for i in range(48000):
        f.write(str(out[i]).replace("\n",""))
        f.write('\n')
    f.close()
