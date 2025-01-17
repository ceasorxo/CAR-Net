import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import scipy.stats
import time

P68 = 78.5
Q = 0.15  #FDR错误发现率


#RSDR:the robust standrad feviation of the residuals
def RSDR(N,k):
    return (P68*N)/(N-k)

# #weighted residuali
# def residual():
#     pass


#RR the distance of a point from the curve divided by the robust standard deviation of the residual 点到曲线的距离除以残差的鲁棒标准值
#RR = [Y-y(x,a0,a1)]/RSDR
def RR():
    pass



def ai(n,i,RSDR):
    ai = Q*(n-(i-1))/n


#t
def T(RSDR):
    return abs()/RSDR



# #两段折线
# def piecewise_linear2(x,x0,k1,k2,cmin1,cmin2):
#     return np.piecewise(x,[x<x0,x>=x0],[lambda x:k1*x+cmin1,
#                                         lambda x:k2*x+cmin2])

#四参数两段折线（x0,k1,k2,cmin）
def piecewise_linear3(x,x0,k1,k2,cmin1):
    return np.piecewise(x,[x<x0,x>=x0],[lambda x:k1*x+cmin1,
                                        lambda x:k2*(x-x0)+(k1*x0+cmin1)])


#特性曲线反函数，通过电容反求受力
def recover_piecewise_linear3(y,x0,k1,k2,cmin1):
    y0 = k1*x0+cmin1
    return np.piecewise(y,[y<y0,y>=y0],[lambda x:(y-cmin1)/k1,
                                        lambda x:((y-(k1*x0+cmin1))/k2)+x0])

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

def draw_curve(data_X,data_Y):
    plt.scatter(data_X, data_Y)
    plt.title("characteristic curve")
    plt.xlabel("F")
    plt.ylabel("C")
    plt.savefig('characteristic curve',format='svg',dpi=300)
    plt.show()

def draw_curve_save(data_X,data_Y,filename):
    plt.scatter(data_X, data_Y)
    plt.title("characteristic curve")
    plt.xlabel("F")
    plt.ylabel("C")
    plt.savefig(filename,format='svg',dpi=300)
    plt.show()

#采用MSE作为目标函数，即适应度
def RMSE_single(data_X,data_Y,p):
    rmse = 0
    for i in range(len(data_X)):
        temp = piecewise_linear3(data_X[i], *p)
        rmse += (data_Y[i]-temp)**2
    rmse = rmse / len(data_X)
    return rmse

#采用MSE作为目标函数，即适应度(整体)
def RMSE(renew_X,renew_Y,p):
    rmse = np.zeros([p.shape[0],1])
    for i in range(p.shape[0]):
        for j in range(len(renew_X)):
            #temp = piecewise_linear(renew_X[j],p[i][0],p[i][1],p[i][2],p[i][3])
            temp = piecewise_linear3(renew_X[j], p[i][0], p[i][1], p[i][2], p[i][3])
            rmse[i] = rmse[i] + (renew_Y[j]-temp)**2
        rmse[i] = rmse[i] / len(renew_X)
    return rmse

class standardpso:
    def __init__(self,fitness_func,renew_X,renew_Y,lower,upper,dim,sizes,max_v=0.6,w=1,c1=2,c2=2,iter_nums=1000,tol=1e-15,ifplot=False,sovmax=False):
        self.fitness_func=fitness_func
        self.lower=lower
        self.upper=upper
        self.dim=dim
        self.sizes=sizes
        self.w=w
        self.c1=c1
        self.c2=c2
        self.iter_num=iter_nums
        self.ifplot=ifplot
        self.sovmax=sovmax
        self.max_v=max_v
        self.tol=tol

    # 速度更新
    def vupdate(self,V,X,pbest,gbest):
        #max v是速度的最大值
        size=V.shape[0]#粒子数量
        r1=np.random.random((size,1))
        r2=np.random.random((size,1))
        V=self.w*V+self.c1*r1*(pbest-X)+self.c2*r2*(gbest-X)
        #注意这一步gbest-x,gbest是一个1*dim的数组，显然和X不同维度。这里用到了numpy矩阵运算的性质简化了公式，本来gbest-X是要写for循环一列一列减的
        #但是numpy直接可以用gbest+-X，意思是先取-X,再用gbest逐行加，最后得到一个和X同维度的东西
        V[V<-self.max_v]=-self.max_v
        V[V>self.max_v]=self.max_v
        return V

    # 位置更新
    def xupdate(self,X,V):
        #X(k+1)=X(k)+tv(k+1)(t=1)
        return X+V

    # 这个func是目标函数
    def fit(self):
        tol2=100#待更新忍耐度
        fitness_value_list=[]#记录种群最优适应度的变化，这个是用来画图的

        # 初始化X
        size=self.sizes
        X=np.zeros((size,self.dim))

        # 初始化pbest和gbest的
        for i in range(0,self.dim):
            X[:,i]=np.random.uniform(self.lower[i],self.upper[i],size=(size))
        V=np.random.uniform(-self.max_v,self.max_v,size=(size,self.dim))
        #第一步操作用于确认pbest和gbest,放到外面
        # p_fitness、g_fitness保存最优值
        p_fitness=self.fitness_func(renew_X,renew_Y,X)#这个返回的是np数组
        g_fitness=p_fitness.min()#获取最小值
        fitness_value_list.append(g_fitness)#群体最优值被认为是当前最优值
        pbest=X
        gbest=X[p_fitness.argmin()]#argmin是寻找最小函数值对应的变量，可以使用它在X中获取索引

        #迭代
        for i in range(1,self.iter_num):
            V=self.vupdate(V,X,pbest,gbest)
            X=self.xupdate(X,V)
            for dimsa in range(self.dim):#变量范围约束
                X1=X[:,dimsa]#这里直接取行
                 #也就是说如果采用索引搜索，和原来的变量占用一个空间
                X1[X1>self.upper[dimsa]]=self.upper[dimsa]#类似于dataframe的搜索操作
                X1[X1<self.lower[dimsa]]=self.lower[dimsa]
            p_fitness2=self.fitness_func(renew_X,renew_Y,X)#这个返回的是np数组
            g_fitness2=p_fitness2.min()#获取最小值，1*1
            #更新最优位置
            for j in range(size):
                if p_fitness[j]>p_fitness2[j]:#P_fitness是函数值，是个一维数组，长为size
                    p_fitness[j]=p_fitness2[j]
                    pbest[j]=X[j]#更新种群最优位置，pbest[j]是位置best。索引与p_fitness对应，从新的X中获取
            if g_fitness>g_fitness2:#群体最优值出现了
                gbest=X[p_fitness2.argmin()]#从群体最优值中获取索引位置
                tol2=g_fitness-g_fitness2
                g_fitness=g_fitness2
            fitness_value_list.append(g_fitness)
            if tol2<self.tol:
                break#两次寻优低于忍耐度，break
        if self.sovmax==True:
            fitness_value_list=-1*np.array(fitness_value_list)

        self.besty=fitness_value_list[-1]
        self.gbest=gbest
        self.fitness_value_list=fitness_value_list

        if self.ifplot==True:
            #画图
            plt.rcParams['font.family'] = ['sans-serif']#防止中文报错
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.plot(fitness_value_list)
            plt.title('迭代过程')

    def getresult(self):
        return self.gbest,self.besty

    def printresult(self):
        print(f"最优变量:{self.gbest}")
        print("最优值是:%5f" % self.besty)

    def getvaluelist(self):
        return self.fitness_value_list



#两种曲线拟合方法性能对比
def method_compare(standrad,popt,best,data_X):
    Y_standrad = piecewise_linear3(data_X, *standrad)
    Y_popt = piecewise_linear3(data_X, *popt)
    Y_pbest = piecewise_linear3(data_X, *best)
    #Y_pso = piecewise_linear2(data_X, pso[0], pso[1], pso[2])

    plt.scatter(data_X, Y_popt,c='blue')
    plt.plot(data_X,Y_popt,c='blue',label='p_popt')

    # plt.scatter(data_X, Y_pso, c='red')
    # plt.plot(data_X, Y_pso, c='red',label='pso')

    plt.scatter(data_X, Y_standrad, c='green')
    plt.plot(data_X, Y_standrad, c='green',label='standrad')

    plt.scatter(data_X, Y_pbest, c='purple')
    plt.plot(data_X, Y_pbest, c='purple', label='p_best')

    plt.title("method-compare")
    plt.xlabel("F")
    plt.ylabel("C")
    plt.legend()
    plt.show()


def method_compare_piecewise_linear3(data_X=None, data_Y=None, popt=None, best=None):
    Y_popt = piecewise_linear3(data_X, *popt)
    Y_pbest = piecewise_linear3(data_X, *best)

    plt.scatter(data_X ,data_Y,c='red')

    #plt.scatter(data_X, Y_popt,c='blue')
    plt.plot(data_X,Y_popt, c='blue', label='Original method of curve fitting')


    #plt.scatter(data_X, Y_pbest, c='purple')
    plt.plot(data_X, Y_pbest, c='green', label='PSO method of curve fitting')

    plt.title("method-compare")
    plt.xlabel("Force")
    plt.ylabel("Capactiance")
    plt.legend()
    plt.savefig('Original method of curve fitting', format='svg', dpi=300)
    plt.show()

def method_compare_point_piecewise_linear3(data_X,data_Y,data_Y_stand,renew_X=[],renew_Y=[],Y_pbest=[],Y_popt=[],renew=0,outliters=0,without_outliters=0):
    plt.scatter(data_X ,data_Y,c='red')
    plt.scatter(data_X, data_Y_stand,c='blue')
    if renew == 1:
        plt.scatter(renew_X,renew_Y,c='green')
    if without_outliters == 1:
        plt.scatter(data_X,Y_popt,c='purple')

    plt.title("method-compare")
    plt.xlabel("F")
    plt.ylabel("C")
    plt.legend()
    plt.show()


def curve_fitting():
    pass


def curve_fitting_with_original_and_pso_method():
    pass

if __name__ == '__main__':
    file_cmin1 = 'cmin1.txt'
    cmin1 = data_loader(file_cmin1)
    cmin1 = np.array(cmin1)
    cmin1 = cmin1.reshape(-1)

    file_k0 = 'k0.txt'
    k0 = data_loader(file_k0)
    k0 = np.array(k0)
    k0 = k0.reshape(-1)
    file_k1 = 'k1.txt'
    k1 = data_loader(file_k1)
    k1 = np.array(k1)
    k1 = k1.reshape(-1)

    file_x0 = 'x0.txt'
    x0 = data_loader(file_x0)
    x0 = np.array(x0)
    x0 = x0.reshape(-1)

    file_route_X = "x.txt"
    file_route_Y = "r_vcmat8x8(200).txt"

    # 数据导入
    data_in_o = data_loader(file_route_X)
    data_out_o = data_loader(file_route_Y)

    print(len(data_in_o), len(data_out_o))
    print(type(data_in_o))
    print(type(data_out_o))
    # print(data_out_o[0][0])

    data_X = data_in_o
    data_X = np.array(data_X).reshape(30000)
    print(data_X.shape)
    data_Y = []

    for i in range(64):
        temp = []
        for j in range(len(data_out_o)):
            temp.append(data_out_o[j][i])
        data_Y.append(temp)

    data_Y = np.array(data_Y)
    print(data_Y.shape)

    #draw_curve(data_X, data_Y[17])

    # #A、曲线预设形式形如piecewise_linear2时
    # for data_num in range(0,1):
    #     print('data_number:',data_num)
    #     p_stand = [float(x0[data_num][0]),float(k0[data_num][0]),float(k1[data_num][0]),float(cmin1[data_num][0]),float(cmin2[data_num][0])]
    #     #单组测试
    #     X = np.zeros(40)
    #     for i in range(40):
    #         X[i] = data_X[i*1000]
    #
    #     temp = data_Y[data_num].reshape(40000)
    #
    #     Y = np.zeros(40)
    #     for j in range(40):
    #         Y[j] = temp[j*1000]
    #
    #     print(X)
    #     print(Y)
    #
    #     draw_curve_save(X,Y,'Distribution of original sampling points')
    #
    #     #1、使用optimize.curve_fit进行初步拟合，利用拟合曲线排除误差点
    #     #popt, pcov = optimize.curve_fit(piecewise_linear, X, Y, bounds=([0, 0, 0, 0], [4., 4., 4., 4.]),maxfev=1000)
    #     popt, pcov = optimize.curve_fit(piecewise_linear2, X, Y, bounds=([0, 0, 0, 0, 0], [4., 4., 4., 4., 4.]),maxfev=1000)
    #     print('popt:',popt)
    #
    #
    #     N = 40
    #     k = 5
    #     residual = np.zeros(N)
    #     for i in range(N):
    #         residual[i] = Y[i] - piecewise_linear2(X[i],popt[0],popt[1],popt[2],popt[3],popt[4])
    #
    #     print('residual:')
    #     print(residual)
    #
    #     residual_sum = 0
    #     for i in range(N):
    #         residual_sum = residual_sum + abs(residual[i])
    #
    #     P68 = (residual_sum/N)*0.6827
    #     print('P68:',P68)
    #
    #
    #     #RSDR
    #     rsdr = RSDR(N, k)
    #     print('RSDR:',rsdr)
    #
    #     #t_ratio
    #     t = np.zeros(N)
    #     for i in range(N):
    #         t[i] = abs(residual[i])/rsdr
    #
    #     print('t_ratio:')
    #     print(t)
    #
    #
    #     p_value = np.zeros(N)
    #     for i in range(N):
    #         p_value[i] = scipy.stats.t.sf(abs(t[i]),df=N-k)*2
    #
    #     print('p_value:')
    #     print(p_value)
    #     p_rank = np.argsort(-p_value)
    #     print('p_rank:')
    #     print(p_rank)
    #     p_value = np.sort(-p_value)
    #     p_value = -p_value
    #     print('p_value:')
    #     print(p_value)
    #
    #     threshold = np.zeros(N)
    #
    #     for i in range(N):
    #         threshold[i] = (Q*(N-(p_rank[i]-1)))/N
    #
    #     print('threshold:')
    #     print(threshold)
    #
    #     out_point = np.zeros(N)
    #
    #     for i in range(N):
    #         if threshold[i]>p_value[i]:
    #             print(threshold[i],p_value[i])
    #             out_point[p_rank[i]] = 1
    #
    #     print('out_point')
    #     print(out_point)
    #
    #     renew_X = []
    #     renew_Y = []
    #     for i in range(len(out_point)):
    #         if out_point[i] == 0:
    #             renew_X.append(X[i])
    #             renew_Y.append(Y[i])
    #     draw_curve_save(renew_X, renew_Y,'Distribution of sampling points with outliers removed')
    #
    #
    #
    #     # #2、循环线性回归拟合曲线
    #     # perr_min = np.inf
    #     # p_best = None
    #     # for n in range(500):
    #     #     p, e = optimize.curve_fit(piecewise_linear2, renew_X, renew_Y, bounds=([0, 0, 0, 0, 0], [4., 4., 4., 4., 4.]),p0=np.random.rand(5) * 4,maxfev=10000)
    #     #     perr = np.sum(np.abs(renew_Y -piecewise_linear2(renew_X, *p)))
    #     #     if (perr < perr_min):
    #     #         perr_min = perr
    #     #         p_best = p
    #     # print('pbest:',p_best)
    #     #
    #     # f = open("parameter_out.txt","a+")
    #     # f.write(str(p_stand))
    #     # f.write(str(popt))
    #     # f.write(str(p_best))
    #     # f.write('\n')
    #     # f.close()
    #
    #
    #
    #
    #
    #     # 3、使用pso算法对排除误差后的采样点进行曲线拟合
    #     # p_low = [0,0,0,0]
    #     p_low = [0, 0, 0, 0, 0]
    #     # p_high = [4,4,1,2]
    #     p_high = [4, 4, 4, 4, 4]
    #     # dim = 4
    #     dim = 5
    #     psize = 500
    #
    #     pso = standardpso(RMSE, renew_X, renew_Y, p_low, p_high, dim, psize)
    #     starttime = time.time()
    #     pso.fit()
    #     endtime = time.time()
    #     print(f"总耗时：{endtime - starttime}s")
    #     pso.printresult()
    #     result = pso.getresult()  # get x y
    #
    #     f = open("pso.txt","a+")
    #     f.write(str(pso.gbest))
    #     f.write('\n')
    #     f.close()


        # # 3、compare-methods
        # file_route_X = "x.txt"
        # file_route_Y = "r_vcmat8x8.txt"
        #
        # # 数据导入
        # data_in_o = data_loader(file_route_X)
        # data_out_o = data_loader(file_route_Y)
        #
        # print(len(data_in_o), len(data_out_o))
        # print(type(data_in_o))
        # print(type(data_out_o))
        # # print(data_out_o[0][0])
        #
        # data_X = data_in_o
        # data_X = np.array(data_X).reshape(40000)
        # print(data_X.shape)
        # data_Y = []
        #
        # for i in range(64):
        #     temp = []
        #     for j in range(len(data_out_o)):
        #         temp.append(data_out_o[j][i])
        #     data_Y.append(temp)
        #
        # data_Y = np.array(data_Y)
        # print(data_Y.shape)
        #
        # #draw_curve(data_X, data_Y[17])
        #
        #
        #
        # #单组测试
        # X = np.zeros(40)
        # for i in range(40):
        #     X[i] = data_X[i*1000]
        #
        # temp = data_Y[0].reshape(40000)
        #
        # Y = np.zeros(40)
        # for j in range(40):
        #     Y[j] = temp[j*1000]
        #
        # print(X)
        # print(Y)
        #
        # # p_least_square, p_new = popt, popt_new
        # #1
        # p_stand = [0.8456714187277494, 2.020257988467175, 0.9846036898420752, 0.09138554404581528, 0.9672087840755959]
        # popt = [1.99999994, 1.47142744, 0.99573486, 0.52877997, 1.12940674]
        # p_best = [0.85473478, 1.58592198, 0.94557937, 0.53200269, 1.2552073 ]
        #
        # # 11
        # # p_standard = [1.5902632819335785,0.31951432957219517,0.7380581873227106,1.2461700936929968]
        # # p_least_square = [1.33148365,0.37843652,0.72533733]
        # # p_pso = [1.44509201,0.35857263,0.72978491]
        #
        # # # 13
        # # p_standard = [0.8140615462176708, 0.224740157298835, 0.6939223587370257]
        # # popt = [0.75539523, 0.25207871, 0.69058008]
        # # p_best = [0.75385829, 0.25186578, 0.69259267]
        #
        #
        # # # 14
        # # p_standard = [2.677714320968603, 0.5838570343739943, 0.1438641432883998]
        # # popt = [2.59789939, 0.59808441, 0.15891558]
        # # p_best = [2.58427167, 0.60024972, 0.15988239]
        #
        # #16
        # #p_standard = [1.2925480732758976,0.7702985842196373,0.2179259580506997]
        # # p_least_square = [2.,0.64011876,0.33934964,1.21606574]
        # # p_pso = [1.10187657,0.77496056,0.23199873,1.21716545]
        # # popt = [1.25097494, 0.80108698, 0.20931834]
        # # p_best = [1.25668725, 0.80004364, 0.21603082]
        #
        #
        # # 17
        # # p_standard = [1.477427962911868, 0.9395618072637323, 0.008021854810055018]
        # # popt = [1.44053584e+00, 9.75366483e-01, 1.35682410e-09]
        # # p_best = [1.43962552, 0.96338239, 0.01694381]
        #
        # # 18
        # # p_standard = [0.3519209922951534, 0.9709304942379522, 0.6435044130448772]
        # # popt = [0.3168379,  1.12015095, 0.62822271]
        # # p_best = [0.31886619, 1.12015095, 0.6282227]
        # # p_pso = [0.31946084, 1.10679261, 0.63100988]
        #
        # # 19
        # # p_standard = [2.3535041583901997, 0.46615890446605696, 0.2113812519747249]
        # # popt = [2.25569888, 0.47151272, 0.23567564]
        # # p_best = [2.29887235, 0.47312131, 0.22971536]
        # # p_pso = [2.31446351, 0.46990514, 0.23165354]
        #
        # # # 21
        # # p_standard = [0.7404796431939069, 0.40857170544182586, 0.7969763036442971]
        # # popt = [0.66876707, 0.46156757, 0.7937976 ]
        # # p_best = [0.6768828, 0.44872409, 0.80300647]
        # # p_pso = [0.68849984, 0.44317192, 0.80221787]
        #
        # # # #22
        # # p_standard = [0.43912166147253445,0.8690434522810627,0.6318294476716529]
        # # # popt = [0.39878,0.99606748, 0.62431278]
        # # # p_best = [0.39536984, 0.99606748, 0.62431278]
        # # # p_pso = [0.38916519, 1.01581837, 0.62250307]
        # #
        # #
        # method_compare(p_stand,popt,p_best,X)
        # # # method_compare(p_standard, popt, p_best, p_pso, X)

    # B、曲线预设形式形如piecewise_linear3
    for data_num in range(13,14):
        print('data_number:', data_num)
        #p_stand = [float(x0[data_num][0]), float(k0[data_num][0]), float(k1[data_num][0]),float(cmin1[data_num][0])]
        p_stand = [float(x0[data_num]), float(k0[data_num]), float(k1[data_num]), float(cmin1[data_num])]
        # 单组测试
        X = np.zeros(60)
        for i in range(60):
            X[i] = data_X[i * 500]

        temp = data_Y[data_num].reshape(30000)

        Y = np.zeros(60)
        for j in range(60):
            Y[j] = temp[j * 500]
            #以采样点前后公11个点的均值作为采样点拟合值
            #Y[j] = (temp[j * 1000 - 5:j * 1000 + 6].sum()) / 11

        print(X)
        print(Y)

        draw_curve_save(X, Y, 'Distribution of original sampling points')

        #直接采用未处理原始点列进行曲线拟合，拟合参数记为popt_first
        popt_first, pcov_first = optimize.curve_fit(piecewise_linear3, X, Y, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),
                                        p0=np.random.rand(4) * 3,
                                        maxfev=1000)
        #print('popt_first:',popt_first)



        # 1、使用optimize.curve_fit进行初步拟合，利用拟合曲线排除误差点
        #对原始等间距采点点列进行平均取样处理，与后续outliers结合消除模型误差
        for j in range(60):
            #以采样点前后公11个点的均值作为采样点拟合值
            Y[j] = (temp[j * 500 - 5:j * 500 + 6].sum()) / 11
            #Y[j] = temp[j * 500]

        draw_curve_save(X, Y, 'Distribution of original sampling points')

        popt, pcov = optimize.curve_fit(piecewise_linear3, X, Y, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),p0=np.random.rand(4) * 3,
                                        maxfev=1000)
        print('popt:', popt)


        N = 60
        k = 4
        residual = np.zeros(N)
        for i in range(N):
            residual[i] = Y[i] - piecewise_linear3(X[i], *popt)

        print('residual:')
        print(residual)

        residual_sum = 0
        for i in range(N):
            residual_sum = residual_sum + abs(residual[i])

        P68 = (residual_sum / N) * 0.6827
        print('P68:', P68)

        # RSDR
        rsdr = RSDR(N, k)
        print('RSDR:', rsdr)

        # t_ratio
        t = np.zeros(N)
        for i in range(N):
            t[i] = abs(residual[i]) / rsdr

        print('t_ratio:')
        print(t)

        p_value = np.zeros(N)
        for i in range(N):
            p_value[i] = scipy.stats.t.sf(abs(t[i]), df=N - k) * 2

        print('p_value:')
        print(p_value)
        p_rank = np.argsort(-p_value)
        print('p_rank:')
        print(p_rank)
        p_value = np.sort(-p_value)
        p_value = -p_value
        print('p_value:')
        print(p_value)

        threshold = np.zeros(N)

        for i in range(N):
            threshold[i] = (Q * (N - (p_rank[i] - 1))) / N

        print('threshold:')
        print(threshold)

        out_point = np.zeros(N)

        for i in range(N):
            if threshold[i] > p_value[i]:
                print(threshold[i], p_value[i])
                out_point[p_rank[i]] = 1

        print('out_point')
        print(out_point)

        renew_X = []
        renew_Y = []
        for i in range(len(out_point)):
            if out_point[i] == 0:
                renew_X.append(X[i])
                renew_Y.append(Y[i])
        draw_curve_save(renew_X, renew_Y, 'Distribution of sampling points with outliers removed.png')
        print(renew_X)
        print(renew_Y)
        #
        # #2、循环线性回归拟合曲线
        # perr_min = np.inf
        # p_best = None
        # for n in range(1000):
        #     p, e = optimize.curve_fit(piecewise_linear3, renew_X, renew_Y, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),p0=np.random.rand(4) * 3,maxfev=1000)
        #     perr = np.sum(np.abs(renew_Y -piecewise_linear3(renew_X, *p)))
        #     if (perr < perr_min):
        #         perr_min = perr
        #         p_best = p
        # print('pbest:',p_best)
        #
        # f = open("parameter_out_piecewise_linear3.txt","a+")
        # f.write(str(p_stand))
        # f.write(str(popt_first))
        # f.write(str(p_best))
        # f.write('\n')
        # f.close()
        #
        # method_compare_piecewise_linear3(renew_X,renew_Y,popt_first,p_best)
        #
        # rmse_popt_first = RMSE_single(X,Y,popt_first)
        # rmse_p_best = RMSE_single(X,Y,p_best)
        # print('RMSE_popt:',rmse_popt_first,'RMSE_p_best',rmse_p_best)
        #
        # rmse = [rmse_popt_first,rmse_p_best]
        # f = open("RMSE_piecewise_linear3.txt","a+")
        # f.write(str(rmse))
        # f.write('\n')
        # f.close()
    #
        # 3、使用pso算法对排除误差后的采样点进行曲线拟合
        p_low = [0, 0, 0, 0]
        p_high = [4, 4, 4, 4]
        # dim = 4
        dim = 4
        psize = 500

        pso = standardpso(RMSE, renew_X, renew_Y, p_low, p_high, dim, psize)
        starttime = time.time()
        pso.fit()
        endtime = time.time()
        print(f"总耗时：{endtime - starttime}s")
        pso.printresult()
        result = pso.getresult()  # get x y
    #     #
    #     # f = open("pso.txt", "a+")
    #     # f.write(str(pso.gbest))
    #     # f.write('\n')
    #     # f.close()

    # #C、随机受力恢复，两种恢复曲线下误差分析
    # file_force = 'force.txt'
    # force = data_loader(file_force)
    # force = np.array(force)
    # #print(force.shape)
    #
    # file_r_vcmat = 'test100_rand/r_vcmat8x8_100.txt'
    # r_vcmat = data_loader(file_r_vcmat)
    #
    # # file_curve_parameter = 'parameter_out_piecewise_linear3.txt'
    # # recover_curve_paremeter = data_loader(file_curve_parameter)
    # # print(len(recover_curve_paremeter))
    #
    # normal_parameter = data_loader('normal_parameter.txt')
    # normal_parameter = np.array(normal_parameter)
    # pso_parmeter = data_loader('pso_parameter.txt')
    # pso_parmeter = np.array(pso_parmeter)

    # # 1、采用两种曲线对受力进行恢复
    # recover_force_without_outliters = np.zeros((100,64))
    # recover_force_with_outliters = np.zeros((100,64))
    #
    # for i in range(len(r_vcmat[0])):
    #     for j in range(len(r_vcmat)):
    #         temp = r_vcmat[j][i]
    #         recover_force_without_outliters[j][i] = recover_piecewise_linear3(temp, normal_parameter[i][0],
    #                                                                                 normal_parameter[i][1],
    #                                                                                 normal_parameter[i][2],
    #                                                                                 normal_parameter[i][3])
    #         # # 对受力恢复到0.1的量级
    #         # recover_force_without_outliters[j][i] = round(recover_force_without_outliters[j][i],1)
    #         if recover_force_without_outliters[j][i]<0:
    #             recover_force_without_outliters[j][i] = 0
    #
    #         recover_force_with_outliters[j][i] = recover_piecewise_linear3(temp, pso_parmeter[i][0],
    #                                                                              pso_parmeter[i][1],
    #                                                                              pso_parmeter[i][2],
    #                                                                              pso_parmeter[i][3])
    #         # # 对受力恢复到0.1的量级
    #         # recover_force_with_outliters[j][i] = round(recover_force_with_outliters[j][i],1)
    #         if recover_force_with_outliters[j][i]<0:
    #             recover_force_with_outliters[j][i]=0
    #
    # recover_force_without_outliters = recover_force_without_outliters.tolist()
    # recover_force_with_outliters = recover_force_with_outliters.tolist()
    #
    # f = open("recover_force_with_orginal_method.txt", "w")
    # for i in range(len(recover_force_without_outliters)):
    #     f.write(str(recover_force_without_outliters[i]).strip('\n'))
    #     f.write('\n')
    # f.close()
    #
    # f = open("recover_force_with_pso_method.txt", "w")
    # for i in range(len(recover_force_without_outliters)):
    #     f.write(str(recover_force_with_outliters[i]).strip('\n'))
    #     f.write('\n')
    # f.close()

    # #2、对比两种曲线下恢复效果
    # recover_force_without_outliters_100 = 'recover_force_with_orginal_method.txt'
    # recover_force_with_outliters_100 = 'recover_force_with_pso_method.txt'
    #
    # recover_force_without_outliters_100 = data_loader(recover_force_without_outliters_100)
    # recover_force_without_outliters_100 = np.array(recover_force_without_outliters_100)
    # recover_force_with_outliters_100 = data_loader(recover_force_with_outliters_100)
    # recover_force_with_outliters_100 = np.array(recover_force_with_outliters_100)
    #
    # error_recover_force_without_outliters = np.zeros((100,64))
    # error_recover_force_with_outliters = np.zeros((100,64))
    #
    # sum_error_recover_force_without_outliters = 0.0
    # error_point_number_in_original_method = 0
    # MSE_error_force_with_original_method = 0.0          #使用 original method 的均方误差
    # sum_error_recover_force_with_outliters = 0.0
    # error_point_number_in_pso_method = 0
    # MSE_error_force_with_pso_method = 0.0               #使用 pso method 的均方误差
    # for i in range(len(force)):
    #     for j in range(len(force[i])):
    #         error_recover_force_without_outliters[i][j] = abs(recover_force_without_outliters_100[i][j]-force[i][j])
    #         error_recover_force_with_outliters[i][j] = abs(recover_force_with_outliters_100[i][j]-force[i][j])
    #         if error_recover_force_without_outliters[i][j]>=0.1 and force[i][j] != 0:
    #             error_point_number_in_original_method += 1
    #             sum_error_recover_force_without_outliters += error_recover_force_without_outliters[i][j]/force[i][j]
    #             MSE_error_force_with_original_method += error_recover_force_without_outliters[i][j]**2
    #         if error_recover_force_with_outliters[i][j]>=0.1 and force[i][j] != 0:
    #             sum_error_recover_force_with_outliters += error_recover_force_with_outliters[i][j]/force[i][j]
    #             error_point_number_in_pso_method += 1
    #             MSE_error_force_with_pso_method += error_recover_force_with_outliters[i][j]**2
    #
    # print('MSE of original method : ' , MSE_error_force_with_original_method/100)
    # print('RMSE of original method : ', math.sqrt(MSE_error_force_with_original_method / 100))
    # print('MSE of pso method : ', MSE_error_force_with_pso_method/100)
    # print('RMSE of pso method : ', math.sqrt(MSE_error_force_with_pso_method / 100))
    #
    #
    #
    # # print('sum_error_recover_force_without_outliters:',
    # #       sum_error_recover_force_without_outliters/error_point_number_in_original_method)
    # print('sum_error_recover_force_without_outliters:',
    #       sum_error_recover_force_without_outliters / (100*64))
    # # print('sum_error_recover_force_with_outliters:',
    # #       sum_error_recover_force_with_outliters/error_point_number_in_pso_method )
    # print('sum_error_recover_force_with_outliters:',
    #       sum_error_recover_force_with_outliters / (100*64))


    # #判断PSO算法中拟合效果不好的是哪些曲线   4 5 11 12 13 14 15 19 26 27 28 29 30 32 33 35 36 38 41 42 45 48 49 50 51 53 55 56 58 59 60 61 62
    # for i in range(len(force[0])):
    #     tmp_error_original = 0.0
    #     tmp_error_pso = 0.0
    #     for j in range(len(force)):
    #         error_recover_force_without_outliters[j][i] = abs(recover_force_without_outliters_100[j][i] - force[j][i])
    #         tmp_error_original += error_recover_force_without_outliters[j][i]/force[j][i]
    #         error_recover_force_with_outliters[j][i] = abs(recover_force_with_outliters_100[j][i] - force[j][i])
    #         tmp_error_pso += error_recover_force_with_outliters[j][i]/force[j][i]
    #     print(tmp_error_original,tmp_error_pso)
    #     if tmp_error_original < tmp_error_pso :
    #         print(i)


    # #f = open("(normal)error_recover_force_without_outliters.txt", "w")
    # f = open("error_recover_force_with_orginal_method.txt", "w")
    # for i in range(len(error_recover_force_without_outliters)):
    #     f.write(str(error_recover_force_without_outliters[i]).strip('\n'))
    #     f.write('\n')
    # f.close()
    #
    # #f = open("(normal)error_recover_force_with_outliters.txt", "w")
    # f = open("error_recover_force_with_pso_method.txt", "w")
    # for i in range(len(error_recover_force_with_outliters)):
    #     f.write(str(error_recover_force_with_outliters[i]).strip('\n'))
    #     f.write('\n')
    # f.close()

    # #D、特性曲线点图绘制，以0.1为最小单位
    # normal_parameter = data_loader("normal_parameter.txt")
    # normal_parameter = np.array(normal_parameter)
    # pso_parmeter = data_loader("pso_parameter.txt")
    # pso_parmeter = np.array(pso_parmeter)
    # RMSE_compare = []
    #
    # Characteristic_point_diagram_without_outliers = []
    # Characteristic_point_diagram_with_outliers = []
    # #4 5 11 12 13 14 15 19 26 27 28 29 30 32 33 35 36 38 4142 45 48 49 50 51 53 55 56 58 59 60 61 62
    # for data_num in range(13,14):
    #     print('data_number:', data_num)
    #     p_stand = [x0[data_num], k0[data_num], k1[data_num],cmin1[data_num]]
    #     print('parameter stand: ',p_stand)
    #     # 单组测试
    #     X = np.zeros(60)
    #     for i in range(60):
    #         X[i] = data_X[i * 500]
    #
    #     temp = data_Y[data_num].reshape(30000)
    #
    #     Y = np.zeros(60)
    #     Y_without_average_sampling = np.zeros(60)
    #     for j in range(60):
    #         Y_without_average_sampling[j] = temp[j * 500]
    #         #Y[j] = (temp[j*1000-2]+temp[j*1000-1]+temp[j*1000]+temp[j*1000+1]+temp[j*1000+2])/5
    #         Y[j] = (temp[j*500-5:j*500+6].sum())/11
    #     # print(X)
    #     # print(Y)
    #
    #     #without_average_sampling linear curve fitting
    #     popt_without_average_sampling, pcov_without_average_sampling = optimize.curve_fit(piecewise_linear3, X, Y_without_average_sampling, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),
    #                                                                                        p0=np.random.rand(4) * 3,
    #                                                                                        maxfev=500)
    #     print('Original method of curve fitting parameter: ', popt_without_average_sampling)
    #
    #     # f = open("normal_parameter.txt","a+")
    #     # f.write(str(popt_without_average_sampling))
    #     # f.write('\n')
    #     # f.close()
    #
    #     method_compare_piecewise_linear3(X, Y_without_average_sampling, popt_without_average_sampling, popt_without_average_sampling)
    #
    #     #draw_curve_save(X, Y, 'Distribution of original sampling points')
    #
    #     #预设点与恢复点对比
    #     Y_stand = np.zeros(60)
    #     for i in range(len(X)):
    #         Y_stand[i] = piecewise_linear3(X[i],x0[data_num],k0[data_num],k1[data_num],cmin1[data_num])
    #     #print(Y_stand)
    #     #method_compare_point_piecewise_linear3(X,Y,Y_stand)
    #
    #
    # #1、使用optimize.curve_fit进行初步拟合，利用拟合曲线排除误差点
    #     popt, pcov = optimize.curve_fit(piecewise_linear3, X, Y, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),p0=np.random.rand(4) * 3,
    #                                     maxfev=1000)
    #     print('popt:', popt)
    #
    #
    #     N = 60
    #     k = 4
    #     residual = np.zeros(N)
    #     for i in range(N):
    #         residual[i] = Y[i] - piecewise_linear3(X[i], *popt)
    #
    #     # print('residual:')
    #     # print(residual)
    #
    #     residual_sum = 0
    #     for i in range(N):
    #         residual_sum = residual_sum + abs(residual[i])
    #
    #     P68 = (residual_sum / N) * 0.6827
    #     # print('P68:', P68)
    #
    #     # RSDR
    #     rsdr = RSDR(N, k)
    #     # print('RSDR:', rsdr)
    #
    #     # t_ratio
    #     t = np.zeros(N)
    #     for i in range(N):
    #         t[i] = abs(residual[i]) / rsdr
    #
    #     # print('t_ratio:')
    #     # print(t)
    #
    #     p_value = np.zeros(N)
    #     for i in range(N):
    #         p_value[i] = scipy.stats.t.sf(abs(t[i]), df=N - k) * 2
    #
    #     # print('p_value:')
    #     # print(p_value)
    #     p_rank = np.argsort(-p_value)
    #     # print('p_rank:')
    #     # print(p_rank)
    #     p_value = np.sort(-p_value)
    #     p_value = -p_value
    #     # print('p_value:')
    #     # print(p_value)
    #
    #     threshold = np.zeros(N)
    #
    #     for i in range(N):
    #         threshold[i] = (Q * (N - (p_rank[i] - 1))) / N
    #
    #     # print('threshold:')
    #     # print(threshold)
    #
    #     out_point = np.zeros(N)
    #
    #     for i in range(N):
    #         if threshold[i] > p_value[i]:
    #             #print(threshold[i], p_value[i])
    #             out_point[p_rank[i]] = 1
    #
    #     print('out_point: ')
    #     print(out_point)
    #
    #     renew_X = []
    #     renew_Y = []
    #     for i in range(len(out_point)):
    #         if out_point[i] == 0:
    #             renew_X.append(X[i])
    #             renew_Y.append(Y[i])
    #     #print('X_axis_with_outliers: ',renew_X)
    #     #print('Y_axis_with_outliers: ',renew_Y)
    #     #draw_curve_save(renew_X, renew_Y, 'Distribution of sampling points with outliers removed.png')
    #     #method_compare_point_piecewise_linear3(X,Y,Y_stand,renew_X,renew_Y,renew = 1)
    #
    #     #2、循环线性回归拟合曲线
    #     perr_min = np.inf
    #     p_best = None
    #     for n in range(1000):
    #         p, e = optimize.curve_fit(piecewise_linear3, renew_X, renew_Y, bounds=([0, 0, 0, 0], [3., 3., 3., 3.]),p0=np.random.rand(4) * 3,maxfev=1000)
    #         perr = np.sum(np.abs(renew_Y -piecewise_linear3(renew_X, *p)))
    #         if (perr < perr_min):
    #             perr_min = perr
    #             p_best = p
    #     print('Recurrent linear regression parameter: ',p_best)
    #     #
    #     # # f = open("parameter_out_piecewise_linear3.txt","a+")
    #     # # f.write(str(p_stand))
    #     # # f.write(str(popt))
    #     # # f.write(str(p_best))
    #     # # f.write('\n')
    #     # # f.close()
    #     method_compare_piecewise_linear3(X, Y_stand, popt_without_average_sampling, p_best)


        # rmse_popt = RMSE_single(X,Y,popt__without_average_sampling)
        # rmse_p_best = RMSE_single(X,Y,p_best)
        # print('RMSE_popt:',rmse_popt,'RMSE_p_best',rmse_p_best)

        # rmse = [rmse_popt,rmse_p_best]
        # f = open("RMSE_piecewise_linear3.txt","a+")
        # f.write(str(rmse))
        # f.write('\n')
        # f.close()

        # # 3、使用pso算法对排除误差后的采样点进行曲线拟合
        # # p_low = [0,0,0,0]
        # p_low = [0, 0, 0, 0]
        # # p_high = [4,4,1,2]
        # p_high = [3, 3, 3, 3]
        # # dim = 4
        # dim = 4
        # psize = 500
        #
        # pso = standardpso(RMSE, renew_X, renew_Y, p_low, p_high, dim, psize)
        # starttime = time.time()
        # pso.fit()
        # endtime = time.time()
        # print(f"总耗时：{endtime - starttime}s")
        # pso.printresult()
        # result = pso.getresult()  # get g_best
        #
        # method_compare_piecewise_linear3(X, Y_stand, popt_without_average_sampling, pso.gbest)



        # f = open("pso_parameter.txt","a+")
        # f.write(str(pso.gbest))
        # f.write('\n')
        # f.close()

    #     #4、对比两种方法差异 1）：使用最小二乘法线性拟合，对采样点无处理  2）：使用PSO算法拟合曲线，采样点进行多点平均和Outlier处理
    #     rmse_normal = RMSE_single(X, Y_stand, normal_parameter[data_num])
    #     rmse_pso = RMSE_single(X, Y_stand, pso_parmeter[data_num])
    #     tmp_rmse = [rmse_normal, rmse_pso]
    #     RMSE_compare.append(tmp_rmse)
    # print('两种方案RMSE对比: ', RMSE_compare)






    #     #4、利用拟合的曲线，复原去除的误差点
    #     recover_X = X
    #     recover_Y = np.zeros(60)
    #     for i in range(len(renew_X)):
    #         index = int(renew_X[i]*20)
    #         recover_Y[index] = renew_Y[i]
    #
    #     for j in range(len(out_point)):
    #         if out_point[j] == 1:
    #             recover_Y[j] = piecewise_linear3(recover_X[j],*p_best)
    #     #method_compare_point_piecewise_linear3(X, Y, Y_stand, recover_X, recover_Y, renew=1)
    #
    #     #拟合后曲线采点与原始点列对比
    #     curve_fit_point_X = X
    #     curve_fit_point_Y = np.zeros(60)
    #     curve_fit_point_Y_outliters = np.zeros(60)
    #     for i in range(len(curve_fit_point_X)):
    #         curve_fit_point_Y[i] = piecewise_linear3(curve_fit_point_X[i], *popt)
    #         curve_fit_point_Y_outliters[i] = piecewise_linear3(curve_fit_point_X[i],*p_best)
    #     method_compare_point_piecewise_linear3(X, Y, Y_stand, renew_X=curve_fit_point_X, renew_Y=curve_fit_point_Y_outliters, Y_popt=curve_fit_point_Y,
    #                                            renew=1,without_outliters=1)


        # Characteristic_point_diagram_without_outliers.append(Y)
        # Characteristic_point_diagram_with_outliers.append(recover_Y)


    # f = open("Characteristic_point_diagram_without_outliers.txt", "w")
    # for i in range(len(Characteristic_point_diagram_without_outliers.append)):
    #     f.write(str(Characteristic_point_diagram_without_outliers.append[i]).strip('\n'))
    #     f.write('\n')
    # f.close()
    #
    # f = open("Characteristic_point_diagram_with_outliers.txt", "w")
    # for i in range(len(Characteristic_point_diagram_with_outliers)):
    #     f.write(str(Characteristic_point_diagram_with_outliers[i]).strip('\n'))
    #     f.write('\n')
    # f.close()



    # #E、根据绘制的特性曲线点图，对受力进行恢复
    # file_force_normal = 'force.txt'
    # force_normal = data_loader(file_force_normal)
































