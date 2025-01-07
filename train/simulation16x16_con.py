import numpy as np
import math

import torch
from torch import nn,optim
from torch.autograd import Variable

import time
from net.CNN_Net import CNN6,CNN9,CNN9_2
import matplotlib.pyplot as plt

def data_loader(file_route):
    file = open(file_route)

    data_load_in = []
    byt = file.readlines()
    #for i in range(len(byt)):
    for i in range(90000):
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
    plt.savefig("image/loss_figure16.png")
    plt.show()



#小批量梯度下降模型训练
def iter_batch_train(data_in,data_out,batch_size,epochs,model_route):

    start = time.perf_counter()

    epoch = 0
    train_loss = []
    epoch_num = []

    num_epochs = math.floor(len(data_in)/batch_size) + 1
    for i in range(epochs):
        #print("_________________________________________________________________________________")
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
            model_route = '../model/model16/onnx/CNN_net9_' + str(epoch+1) + '.pt'
            torch.save(model.state_dict(), model_route)
    torch.save(model.state_dict(), model_route)
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
    epochs = 5
    batch_size = 128
    model_route = 'model/model16/CNN_net9_100.pt'

    file_route_in = "../data/sim16/cmeans16x16.txt"
    file_route_out = "../data/sim16/vcmat16x16.txt"

    # 训练数据导入
    data_in_o = data_loader(file_route_in)
    data_in_o = np.array(data_in_o)
    data_in = np.zeros((90000,16,16),dtype=np.float32)
    for i in range(len(data_in_o)):
        data_in[i] = data_in_o[i].reshape(16,16)
    data_in = data_in.astype(np.float32)

    data_out_o = data_loader(file_route_out)
    data_out_o = np.array(data_out_o)
    data_out = np.zeros((90000,16,16),dtype=np.float32)
    for i in range(len(data_out_o)):
        data_out[i] = data_out_o[i].reshape(16,16)
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
    model = CNN9().float()
    #model.load_state_dict(torch.load(model_route))
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