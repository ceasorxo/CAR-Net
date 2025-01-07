import numpy as np
import math

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

import time
from net.CNN_Net import CNN5_8,CNN5,CNN4_1,CNN7_1,CNN11,CNN11_1,VGG8_32, ResNet18_8, DnCNN, CNN11_2
import matplotlib.pyplot as plt

def data_loader(file_route):
    file = open(file_route)

    data_load_in = []
    byt = file.readlines()
    #for i in range(len(byt)):
    for i in range(0,40000):
        temp = byt[i].strip()
        temp = temp.split(' ')
        temp = np.array(temp)
        data = [float(i) for i in temp]
        data_load_in.append(data)
    return data_load_in


#loss图像
def loss_figure(epoch_num,train_loss,model_route):
    plt.figure()
    plt.title(model_route)
    plt.plot(epoch_num,train_loss,label='loss-curve')
    plt.ylabel('Loss')
    plt.xlabel('epoch_num')
    plt.legend()
    plt.savefig("image/loss_figure.png")
    plt.show()

#自定义损失函数，各分段数据损失权值一致
def my_mse_loss(x,y):
    return torch.mean(torch.pow(((x-y)/y),2)+torch.pow((x-y),2))


def iter_batch_train(data_in,data_out,batch_size,epochs,model_route):

    start = time.perf_counter()

    epoch = 0
    train_loss = []
    epoch_num = []

    num_epochs = math.floor(len(data_in)/batch_size) + 1
    for i in range(epochs):
        print("_________________________________________________________________________________")
        loss_total = 0
        # rng = np.random.default_rng(i)
        # rng.shuffle(data_in)
        # rng = np.random.default_rng(i)
        # rng.shuffle(data_out)
        for j in range(num_epochs):
            if j == (num_epochs - 1):
                data_in_batch = data_in[j * batch_size:]
                np.random.shuffle(data_in_batch)
                data_out_batch = data_out[j * batch_size:]
                np.random.shuffle(data_out_batch)
            else:
                data_in_batch = data_in[j*batch_size : (j+1)*batch_size]
                np.random.shuffle(data_in_batch)
                data_out_batch = data_out[j*batch_size : (j+1)*batch_size]
                np.random.shuffle(data_out_batch)

            data_in_batch = Variable(data_in_batch).cuda()
            data_out_batch = Variable(data_out_batch).cuda()
            out = model(data_in_batch)
            loss = criterion(out, data_out_batch)
            loss_total += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1

        if epoch % 1 == 0:
            print('epoch:{},loss:{:.16}'.format(epoch, loss.data.item()))
            print('loss_total:{:.16}'.format((loss_total/num_epochs)))
            train_loss.append(loss.data.item())
            epoch_num.append(epoch)
        if ((epoch+1) % 1) == 0:
            model_route = '../model/model32/onnx/CNN_net11-' + str(epoch+1) + '.pt'
            torch.save(model.state_dict(), model_route)
    #torch.save(model.state_dict(), model_route)
    end = time.perf_counter()
    print(end - start)
    loss_figure(epoch_num, train_loss, model_route)


#将数据放大k倍
def data_enlarge(data,k):
    for i in range(len(data)):
        temp = data[i]
        for j in range(len(temp)):
            data[i][j] = data[i][j]*k
    return data

if __name__ == "__main__":
    # 学习率、迭代次数、训练批次大小
    learning_rate = 0.005
    epochs = 100
    batch_size = 128
    model_route = '../model/model32/onnx/CNN_net11.pt'

    #file_route_in = "../data/sim8/1-2/cmeans8x8.txt"     # 200000
    #file_route_in = "../data/sim16/cmeans16x16.txt"  # 100000
    file_route_in = "../data/sim32/cmeans32x32.txt"  # 48000
    #file_route_out = "../data/sim8/1-2/vcmat8x8.txt"
    #file_route_out = "../data/sim16/vcmat16x16.txt"
    file_route_out = "../data/sim32/vcmat32x32.txt"

    # 训练数据导入
    data_in_o = data_loader(file_route_in)
    data_in_o = np.array(data_in_o)
    data_in = np.zeros((40000,32,32),dtype=np.float32)
    for i in range(len(data_in_o)):
        data_in[i] = data_in_o[i].reshape(32,32)
    data_in = data_in.astype(np.float32)
    
    data_out_o = data_loader(file_route_out)
    data_out_o = np.array(data_out_o)
    data_out = np.zeros((40000,32,32),dtype=np.float32)
    for i in range(len(data_out_o)):
        data_out[i] = data_out_o[i].reshape(32,32)
    data_out = data_out.astype(np.float32)
    
    print(len(data_in), len(data_out))
    print(data_in.shape)
    print(data_out.shape)
    print(data_out.dtype)
    
    #print(data_in[0])
    
    data_in = data_enlarge(data_in,1000000)
    data_out = data_enlarge(data_out,1000000)
    
    # 数据类型转换
    data_in = torch.tensor(data_in)
    data_in = torch.unsqueeze(data_in, dim=1)
    data_out = torch.tensor(data_out)
    data_out = torch.unsqueeze(data_out, dim=1)
    
    print(data_in.dtype)
    print(data_out.dtype)
    print(data_in.size())
    print(data_out.size())
    
    
    # 模型选择
    model = CNN11().float()
    model.train()
    # 'CNN_net5_drpout0.3(8x8x1-(3x1x32+1x3x32+1x5x32+5x1x32)x7-8x8x32-8x8x1-1000-0.005-batchsize=128)'
    # model.load_state_dict(torch.load(
    #     '../model/model8/vg8/VGG8(100-0.005-batchsize=128).pt'))
    #print(model)
    #summary(model,input_size=(batch_size,1,8,8))
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 损失函数与优化器
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 小批量梯度下降模型
    iter_batch_train(data_in, data_out, batch_size, epochs, model_route)