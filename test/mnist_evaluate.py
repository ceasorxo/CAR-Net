import torch
from net.CNN_Net import ResNet
from torch.autograd import Variable
import numpy as np
#import xlwt

def data_loader(file_route):
    file = open(file_route)

    data_load_in = []
    # 读取一行数据
    byt = file.readlines()
    for i in range(len(byt)):
        #去每一行末尾的\n
        temp = byt[i].strip()
        temp = temp.split(' ')
        #转化为数组
        temp = np.array(temp)
        data = [float(i) for i in temp]
        data_load_in.append(data)
    return data_load_in


if __name__ == '__main__':

    model_route = 'model/model32/mnist/48000/CNN(250)/ResNet/cmeans_ResNet(100-0.001-batchsize=128).pt'
    #model_route = 'model/model32/mnist/48000/CNN(250)/LeNet2/cmeans_LeNet2(1000-0.00001-batchsize=128).pt'

    file_route_in = "../data/sim32p/48000/cmeans32x32.txt"
    #file_route_out = "data/sim32p/48000/label32x32.txt"


    #训练数据导入
    data_in_o = data_loader(file_route_in)
    data_in_o = np.array(data_in_o)
    data_in = np.zeros((48000, 32, 32), dtype=np.float64)
    for i in range(len(data_in_o)):
        data_in[i] = data_in_o[i].reshape(32, 32)
    data_in = data_in.astype(np.float64)

    # data_out_o = data_loader(file_route_out)
    # data_out_o = np.array(data_out_o)
    # data_out = data_out_o.astype(np.float64)

    print(data_in.shape)
    # print(data_out.shape)
    # print(data_out.dtype)


    # 数据类型转换
    data_in = torch.tensor(data_in)
    data_in = torch.unsqueeze(data_in, dim=1)
    # data_out = torch.tensor(data_out)
    # data_out = torch.unsqueeze(data_out, dim=1)
    # data_out = torch.unsqueeze(data_out, dim=1)

    print(data_in.dtype)
    print(data_in.size())
    # print(data_out.dtype)
    # print(data_out.size())

    # 导入模型
    model = ResNet().double()
    #model = model.cuda()
    model.load_state_dict(torch.load(model_route))


    out = []
    for i in range(2):
        temp = data_in[i*100:(i+1)*100]
        temp = Variable(temp)
        out_temp = model(temp)
        out_temp = out_temp.detach().numpy()
        out.append(out_temp)
    out = np.array(out)


    # data_in = Variable(data_in)
    # # data_out = Variable(data_out)
    # out = model(data_in[0:100])
    # #out = abs(out *1000000)

    out = np.reshape(1,200,10)
    out = np.squeeze()

    #out = out.detach().numpy()
    #out = out.numpy()



    f = open("../out.txt", "a")
    for i in range(len(out)):
        f.write(str(out[i]).strip('[').strip(']'))
        f.write('\n')
    f.close()


    out = out.detach().numpy()
    label = np.argmax(out,axis=1)
    count = 0
    for i in range(10):
        for j in range(4800):
            if(label[i*4800+j] == i):
                count = count +1
    print(count/48000)