# Motion Sense (SJTU, EI327)

本项目是上海交通大学 *EI327-1-工程实践与科技创新 IV-I* 课程项目。

## 简介

通过收集和分析各类运动传感器（陀螺仪、加速度计）上的数据，我们可以得到目标对象
的运动信息。本项目的中的任务是**根据加速度计与陀螺仪数据进行运动状态分类**。

本项目基于[Motion Sense数据集](https://github.com/mmalekzadeh/motion-sense)。主要工作分为已下三个方面：

1. 复现论文[Deep learning analysis of mobile physiological, environmental and location sensor data for emotion detection - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253518300460)中提出的CNN+LSTM模型，并比较 CNN 和 LSTM 模型的时序表达能力。  
2. 尝试了多种不同的机器学习模型在运动检测方面的效果，达到了远超 CNN+LSTM方法的准确率 (**97.8%**)。
3. 探究 Motion Sense数据集的**数据内在特性**（1. 单个输入包含的时间点数量、 2. 数据有效率、 3. 数据集中性、 4. 传感器重要程度）。  

## 目录结构

```
T:.
├─Results of ensemble learning			// 工作2的原始结果
├─Slides								// 终期答辩、4次讨论的PPT
└─Source Code							// 源代码
    ├─Baseline							// CNN+LSTM模型（Pytorch Templ
    │  ├─base
    │  ├─data							// 原始数据（遵循pytorch-template: https://github.com/victoresque/pytorch-template）
    │  │  ├─A_DeviceMotion_data
    │  ├─data_loader
    │  ├─logger
    │  ├─model
    │  ├─saved
    │  ├─trainer
    │  └─utils
    └─TraditionalML						// 集成学习模型（Autogluon: https://auto.gluon.ai/）
```

## 使用

### 1. 数据预处理

数据预处理文件为`Source Code/Baseline/data_preprocess.py`

### 2. 模型训练

**CNN+LSTM**: 代码结构遵循[pytorch-template](https://github.com/victoresque/pytorch-template)，使用方式可点击链接查看项目主页。

**集成学习模型**: 使用[Autogluon](https://auto.gluon.ai/)，代码文件为`Source Code/TraditionML/MachineLearning.py`。

## 实验设定与结果

### 设定

![图片7](https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%877.png)

### 结果

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%871.png" alt="图片1" style="zoom: 50%;" />

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%872.png" alt="图片2" style="zoom: 50%;" />

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%873.png" alt="图片3" style="zoom:50%;" />

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%874.png" alt="图片4" style="zoom:50%;" />

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%875.png" alt="图片5" style="zoom:50%;" />

<img src="https://michael-picgo.obs.cn-east-3.myhuaweicloud.com/%E5%9B%BE%E7%89%876.png" alt="图片6" style="zoom:50%;" />
