import random
import torch
from net.CNN_Net import CNN11,CNN5,CNN4_1,CNN7_1,VGG8_8,ResNet18_8,DnCNN,CNN11_1
from torch.autograd import Variable
import numpy as np
#import xlwt
import matplotlib.pyplot as plt
from scipy.stats import norm,t


def write_result_to_excel_single(data_out,out,error,savepath):
    myWorkbook = xlwt.Workbook(encoding='utf-8')
    mySheet = myWorkbook.add_sheet('Sheet1')
    mySheet.write(0, 0, label='实际输出')
    mySheet.write(0, 1, label='模型输出')
    mySheet.write(0, 2, label='误差')
    mySheet.write(0, 3, label='误差百分比(绝对值)')

    data_out = data_out.detach().numpy()
    data_out = np.squeeze(data_out)
    out = out.detach().numpy()
    out = np.squeeze(out)
    error = error.detach().numpy()
    error = np.squeeze(error)
    # print(data_out.shape)
    # print(out.shape)
    # print(error.shape)

    sum = 0
    num = 0
    num_out = 0

    for i in range(data_out.shape[0]):
        h= data_out.shape[1]
        for j in range(data_out.shape[1]):
            mySheet.write(i*h + j + 1, 0, label=str(data_out[i][j]))
            mySheet.write(i*h + j + 1, 1, label=str(out[i][j]))
            mySheet.write(i*h + j + 1, 2, label=str(error[i][j]))
            mySheet.write(i*h + j + 1, 3, label=str(abs(error[i][j] / data_out[i][j])))
            sum = sum +abs(error[i][j] / data_out[i][j])
            # if data_out[i][j][k] == 1.0:
            #     num = num + 1
            #     if abs(error[i][j][k] / data_out[i][j][k])<=0.01:
            #         num_out = num_out + 1

    myWorkbook.save(savepath)


def write_result_to_excel(data_out,out,error,savepath):
    myWorkbook = xlwt.Workbook(encoding='utf-8')
    mySheet = myWorkbook.add_sheet('Sheet1')
    mySheet.write(0, 0, label='实际输出')
    mySheet.write(0, 1, label='模型输出')
    mySheet.write(0, 2, label='误差')
    mySheet.write(0, 3, label='误差百分比(绝对值)')

    data_out = data_out.detach().numpy()
    data_out = np.squeeze(data_out)
    out = out.detach().numpy()
    out = np.squeeze(out)
    error = error.detach().numpy()
    error = np.squeeze(error)
    # print(data_out.shape)
    # print(out.shape)
    # print(error.shape)

    sum = 0
    num = 0
    num_out = 0

    for i in range(data_out.shape[0]):
        h= data_out.shape[1]
        w = data_out.shape[2]
        tmp = h*w
        for j in range(data_out.shape[1]):
            for k in range(data_out.shape[2]):
                mySheet.write(i*tmp + j*h + k + 1, 0, label=str(data_out[i][j][k]))
                mySheet.write(i*tmp + j*h + k + 1, 1, label=str(out[i][j][k]))
                mySheet.write(i*tmp + j*h + k + 1, 2, label=str(error[i][j][k]))
                mySheet.write(i*tmp + j*h + k + 1, 3, label=str(abs(error[i][j][k] / data_out[i][j][k])))
                sum = sum +abs(error[i][j][k] / data_out[i][j][k])
                # if data_out[i][j][k] == 1.0:
                #     num = num + 1
                #     if abs(error[i][j][k] / data_out[i][j][k])<=0.01:
                #         num_out = num_out + 1
    myWorkbook.save(savepath)


def data_loader(file_route):
    file = open(file_route)

    data_load_in = []
    byt = file.readlines()
    #for i in range(len(byt)):
    #for i in range(90000,100000):
    for i in range(40000,48000):
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

#将数据归一化到1-2的范围以适应训练的模型，x[i] = x[i]/(max-min)+1
def data_normalization(data,max=23,min=0):
    for i in range(len(data)):
        temp = data[i]
        for j in range(len(temp)):
            data[i][j] = data[i][j]/(max-min)
    return data

#将归一化后的数据复原到原始数据范围，y[i] = y[i]*(max-min)
def data_recovery(data,max=23,min=0):
    for i in range(len(data)):
        temp = data[i]
        for j in range(len(temp)):
            data[i][j] = data[i][j]*(max-min)
    return data

#小批量验证集
def iter_batch_test(data_in,data_out,batch_size):
    start = random.randint(0, len(data_in)-batch_size)
    test_data_in = data_in[start:start+batch_size]
    test_data_out = data_out[start:start+batch_size]
    return test_data_in,test_data_out

#训练数据处理
def data_cleaning(data_in,data_out):
    tmp = []
    for i in range(len(data_out)):
        if(data_out[i][0] < 0.2 or data_out[i][0] > 0.8):
            tmp.append(i)
    data_out = np.delete(data_out,tmp,0)
    data_in = np.delete(data_in,tmp,0)
    return data_in,data_out


#小批量验证集
def iter_batch_test(data_in,data_out,batch_size):
    start = random.randint(0, len(data_in)-batch_size)
    test_data_in = data_in[start:start+batch_size]
    test_data_out = data_out[start:start+batch_size]
    return test_data_in,test_data_out

def error_distribution(error=None):
    df = 10
    error = error.reshape((-1))
    #print(error.shape)
    max = error.max()
    min = error.min()
    arrange = max - min
    error = error / (arrange/2) # 误差归一化到[-1，1]

    n,bins = np.histogram(error,bins=50,normed=True,density=True)
    print(n,bins)

    weights = np.ones_like(error)/float(len(error))
    plt.hist(error,bins=bins,weights=weights,alpha=1,edgecolor='white',label='model recovery error distribution')
    plt.title('Error_Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # x = np.linspace(-1,1,100)
    # pdf = t.pdf(x,df)
    # plt.plot(x,pdf,label=f"t-probability distribution(df={df})")
    plt.legend()
    plt.savefig('error_distribution.svg', format='svg', dpi=300)
    plt.show()

def error_normalize(error):
    error = error.reshape((-1))

    max = error.max()
    min = error.min()
    arrange = max - min
    error = error/(arrange/2)    #误差归一化到[-1，1]

    error_interval = [-1+i*0.04 for i in range(51)]

    error_interval_num = np.zeros((50))
    for i in range(len(error)):
        index = int((error[i]-(-1))/0.04)
        error_interval_num[index] += 1
    error_interval_num /= len(error)
    return error_interval,error_interval_num





if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)  # 全部输出
    batch_size = 128
    #real测试数据集
    #file_route_in = "data/sim5x10/test/10000/cmeans5x10.txt"
    #file_route_in = "data/sim5x10/train/10000/cmeans5x10.txt"
    #file_route_in = "../data/sim8/1-2/cmeans8x8.txt"
    #file_route_in = "data/sim8/1-5/data/cmeans8x8.txt"
    #file_route_in = "data/sim8/1-5/1.5-4.5/cmeans8x8(1.5-4.5).txt"
    #file_route_in = "data/sim8g/parameter-final/data/cmeans8x8g.txt"
    #file_route_in = "../data/sim16/cmeans16x16.txt"
    #file_route_in = "data/sim16p/cmeans16x16.txt"
    file_route_in = "../data/sim32/cmeans32x32.txt"
    #file_route_in = "data/sim32p/48000/cmeans32x32.txt"
    #file_route_in = "../data/sim32p/60000/test/cmeans32x32.txt"
    #file_route_out = "data/sim5x10/test/10000/vcmat5x10.txt"
    #file_route_out = "data/sim5x10/train/10000/vcmat5x10.txt"
    #file_route_out = "../data/sim8/1-2/vcmat8x8.txt"
    #file_route_out = "data/sim8/1-5/data/vcmat8x8.txt"
    #file_route_out = "data/sim8/1-5/1.5-4.5/vcmat8x8(1.5-4.5).txt"
    #file_route_out = "data/sim8g/parameter-final/data/vcmat8x8g.txt"
    #file_route_out = "../data/sim16/vcmat16x16.txt"
    #file_route_out = "data/sim16p/vcmat16x16.txt"
    file_route_out = "../data/sim32/vcmat32x32.txt"
    #file_route_out = "data/sim32p/48000/vcmat32x32.txt"
    #file_route_out = "../data/sim32p/60000/test/vcmat32x32.txt"


    #训练数据导入
    data_in_o = data_loader(file_route_in)
    data_in_o = np.array(data_in_o)
    data_in = np.zeros((8000, 32, 32), dtype=np.float64)
    for i in range(len(data_in_o)):
        data_in[i] = data_in_o[i].reshape(32, 32)
    data_in = data_in.astype(np.float64)

    data_out_o = data_loader(file_route_out)
    data_out_o = np.array(data_out_o)
    data_out = np.zeros((8000, 32, 32), dtype=np.float64)
    for i in range(len(data_out_o)):
        data_out[i] = data_out_o[i].reshape(32, 32)
    data_out = data_out.astype(np.float64)

    print(len(data_in), len(data_out))
    print(data_in.shape)
    print(data_out.shape)
    print(data_out.dtype)



    # ####################################################################################################################
    # #64-n-64
    # #训练数据导入
    # data_in_o = data_loader(file_route_in)
    # data_in_o = np.array(data_in_o)
    # data_in = data_in_o.astype(np.float64)
    #
    # data_out_o = data_loader(file_route_out)
    # data_out_o = np.array(data_out_o)
    # data_out = data_out_o.astype(np.float64)
    #
    # print(len(data_in), len(data_out))
    # print(data_in.shape)
    # print(data_out.shape)
    # print(data_out.dtype)
    # ####################################################################################################################


    data_in = data_enlarge(data_in, 1000000)
    data_out = data_enlarge(data_out, 1000000)


    # ####################################################################################################################
    # #1、部分数据测试
    #
    #
    # #选取生成测试集
    # test_data_in,test_data_out = iter_batch_test(data_in,data_out,128)
    # print(len(test_data_in),len(test_data_out))
    #
    # # 数据类型转换
    # test_data_in = torch.tensor(test_data_in)
    # test_data_in = torch.unsqueeze(test_data_in, dim=1)
    # test_data_out = torch.tensor(test_data_out)
    # test_data_out = torch.unsqueeze(test_data_out, dim=1)
    #
    # print(test_data_in.dtype)
    # print(test_data_out.dtype)
    # print(test_data_in.size())
    # print(test_data_out.size())
    #
    # # # 损失函数
    # # criterion = nn.MSELoss()
    # #
    # # # 模型评估
    # # eval_loss = 0
    # # eval_acc = 0
    #
    # # data_in = torch.tensor(data_in)
    # # data_in = torch.unsqueeze(data_in, dim=1)
    #
    # # data_out = []
    #
    # # 导入模型
    # model = CNN5_8().double()
    # model.load_state_dict(torch.load('model/model8/1-5/my_loss/normal_RMSE/CNN_net5_8_drpout0.3(8x8x1-(3x1x32+1x3x32+1x5x32+5x1x32)x8-8x8x32-8x8x1-2300-0.001-batchsize=128).pt',map_location='cuda:0'))
    # # model.load_state_dict(torch.load(
    # #     'model/model32/CNN/0/CNN_net11_Drop0.3(32x32x1-(3x1+1x3+1x5+5x1)x16-32x32x128-50-0.005-batchsize=64).pt',
    # #     map_location='cuda:0'))
    # #model.eval()
    #
    # # #data_in = Variable(data_in)
    # # for i in range(1000):
    # #     out = model(data_in[i*10:i*10+10])
    # #     out = out/1000000
    # #     out = torch.squeeze(out, dim=1)
    # #     temp = out.detach().numpy()
    # #     #print(temp.shape)
    # #     data_out.append(temp)
    #
    # # data_out = np.array(data_out)
    # # print(data_out.shape)
    # # out = np.zeros((10000, 256), dtype=np.float64)
    # # for i in range(1000):
    # #     for j in range(10):
    # #         for k in range(16):
    # #             for m in range(16):
    # #                 out[i*10+j,k*16+m] = data_out[i][j][k][m]
    # #
    # # print(out.shape)
    # # print(str(out[0]))
    #
    #
    #
    # # f = open("r_vcmat16x16.txt", "w")
    # # for i in range(10000):
    # #     f.write(str(out[i]))
    # #     f.write('\n')
    # # f.close()
    #
    #
    # #out = model(data_in)
    #
    #
    #
    #
    # test_data_in = Variable(test_data_in)
    # test_data_out = Variable(test_data_out)
    # out = model(test_data_in)
    # real_error = test_data_out - out
    # print(test_data_out.size())
    # print(out.size())
    # print(real_error.size())
    # # print("实际输出：{} ".format(test_data_out))
    # # print("输出：{} ".format(out))
    # # print("误差：{} ".format(real_error))
    #
    #
    # write_result_to_excel(test_data_out/1000000, out/1000000, real_error/1000000, savepath='model/model8/1-5/my_loss/normal_RMSE/CNN_net5_8_drpout0.3(8x8x1-(3x1x32+1x3x32+1x5x32+5x1x32)x8-8x8x32-8x8x1-2300-0.001-batchsize=128).xls')

    ####################################################################################################################
    #2、整体误差测试
    data_in = torch.tensor(data_in)
    data_in = torch.unsqueeze(data_in, dim=1)
    data_out = torch.tensor(data_out)
    data_out = torch.unsqueeze(data_out, dim=1)

    #model = ResNet18_8().double()
    model = CNN11_1().double()
    model.load_state_dict(torch.load(
        '../model/model32/googlenet/CNN_net11_1_drpout0.3(8x8x1-(1x1+3x3+5x5+3x3)x16x32-8x8x32-8x8x1-30-0.005-batchsize=128).pt',
        map_location='cuda:0'))
    model.eval()

    sum = 0
    for i in range(80):
        temp_in = data_in[i*100:(i+1)*100]
        # # # 为使新数据适应模型，对数据做归一化处理
        # temp_in = data_normalization(temp_in)
        temp_in = Variable(temp_in)
        temp_out = data_out[i * 100:(i + 1) * 100]
        temp_out = Variable(temp_out)
        #temp_out = torch.reshape(temp_out,(100,8,8))
        #print(temp_out.shape)

        out = model(temp_in)
        #print(out.shape)

        # #根据cmeans放大变换比例修正vcmat的值
        # out = data_recovery(out)


        real_error = out - temp_out
        #print(real_error)
        #print(real_error.size())

        error_save = real_error.detach().numpy()
        error_save = error_save.reshape((100,-1))
        #print(error_save.shape)


        temp_out = temp_out.detach().numpy()
        temp_out = np.squeeze(temp_out)
        real_error = real_error.detach().numpy()
        real_error = np.squeeze(real_error)
        # print(temp_out.shape)
        # print(real_error.shape)

        for m in range(100):
            for n in range(32):
                for k in range(32):
                    sum  = sum + abs(real_error[m][n][k] / temp_out[m][n][k])

    ave_sum = sum/(8000*32*32)
    print(ave_sum)

    #error_interval,error_interval_num = error_normalize(error_save)
    #error_distribution(error=error_save)


    # #(64-n-64)
    # ####################################################################################################################
    # # 整体误差测试
    # data_in = torch.tensor(data_in)
    # data_in = torch.unsqueeze(data_in, dim=1)
    # data_out = torch.tensor(data_out)
    # data_out = torch.unsqueeze(data_out, dim=1)
    #
    # model = CNN().double()
    # model.load_state_dict(torch.load(
    #     'model/model8/mlp/CNNS/model8/CNNS/CNN(8x8x1-(3x3x64)-8x8x64-8x8x32-8X8X1-400-0.005-batchsize=128).pt',
    #     map_location='cuda:0'))
    #
    # sum = 0
    # for i in range(100):
    #     temp_in = data_in[i * 100:(i + 1) * 100]
    #     temp_in = Variable(temp_in)
    #     temp_out = data_out[i * 100:(i + 1) * 100]
    #     temp_out = Variable(temp_out)
    #
    #     out = model(temp_in)
    #     real_error = out - temp_out
    #     # print(real_error.size())
    #
    #     temp_out = temp_out.detach().numpy()
    #     temp_out = np.squeeze(temp_out)
    #     real_error = real_error.detach().numpy()
    #     real_error = np.squeeze(real_error)
    #     # print(temp_out.shape)
    #     # print(real_error.shape)
    #
    #     for i in range(100):
    #         for j in range(64):
    #                 sum = sum + abs(real_error[i][j] / temp_out[i][j])
    #
    # ave_sum = sum / (10000 * 8 * 8)
    # print(ave_sum)


    #模型计算量与参数量参数
    input = torch.randn(1,1,32,32)
    model = CNN11()
    # model.load_state_dict(torch.load('CNN_net4_Dropout0.2(8x8x1-(3x1x32+1x3x32+1x5x32+5x1x32)x4-8x8x32-8x8x1-1000-0.005-batchsize=128).pt',map_location='cuda:0'))
    flops, params = profile(model, inputs=(input,))
    print(flops,params)
