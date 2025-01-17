# CAR-Net

##CAR-Net python
- data
	- sim8
	- sim8g
	- sim16
	- sim16p
	- sim32
	- sim32p
- model
	- model8
	- model16
	- model32
- net
	- Net.py
- test
	- evaluate_con.py
	- inference_time.py
	- mnist_evaluate.py
	- mnist_recognition.py
- train
	- simulation8x8_con.py
	- simulation16x16_con.py
	- simulation32x32_con.py
- experiment
	- data_caculate.py
	- model_out.py
	- ROUT.py
	- pt_to_onnx.py

###data
Trainning dataset generated by SPICE for CAR-Net model，corresponding to flexible capacitive sensor arrays with sizes of 8x8, 16x16, 32x32.  
sim8、sim16、sim32: Trainning dataset for flexible capacitive sensor arrays with sizes of 8x8, 16x16, 32x32, respectively.  
sim8g：Measured dataset for actual sensor array.  
sim16p、sim32p：Capacitor crosstalk simulation of mnist image dataset.  

----------

CAR-Net训练数据集，由SPICE仿真生成，对应尺寸8x8、16x16、32x32大小的柔性电容式传感阵列。  
sim8、sim16、sim32：对应尺寸8x8、16x16、32x32大小的柔性电容式传感阵列仿真训练集  
sim8g：阵列实测数据集  
sim16p、sim32p：mnist图像数据集电容串扰仿真  

----------

Link: [https://pan.baidu.com/s/1_aavx19Iy2tjK6mcHY2h7g ](https://pan.baidu.com/s/1_aavx19Iy2tjK6mcHY2h7g )  
Code: carn  

###model  
model8、model16、model32：CAR-Net model for flexible capacitive sensor arrays with sizes of 8x8, 16x16, 32x32.  

----------

model8、model16、model32：尺寸8x8、16x16、32x32大小的柔性电容式传感阵列所对应的CAR-Net模型

###net
Net: Model structure for CAR-Net model of diferent sensor arrays sizes.

----------

Net：CAR-Net model 模型结构

###train
Train CAR-Net model for diferent sensor arrays sizes.

----------

CAR-Net模型训练

###test
Model performance and model inference speed for CAR-Net model.  
Experimental results of mnist image dataset for CAR-Net model.

----------

CAR-Net模型性能测试、模型推理速度测试  
mnist图像数据集上CAR-Net model恢复效果与不同基础图像分类网络分类性能测试

###experiment
data caculate: Capacitor crosstalk analysis of 3x3 capacitive sensor array.  
mdoel out: Output for CAR-Net model.    
pt to onnx: Model transformation.  
ROUT: Outlier removal algorithm.  

----------

data caculate:3x3规格电容传感阵列电容串扰分析  
mdoel out:模型结果输出  
pt to onnx:模型转换  
ROUT:CAR-Net model 误差点去除算法  


##Notes
When using our dataset, please cite the source of the reference.  

    CAR-Net: Solving Electrical Crosstalk Problem in Capacitive Sensing Array. [ASP-DAC 2025]

----------
  
使用本项目数据集请标注引用来源。