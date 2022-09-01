# CNN-FPGA

基于FPGA的cifar-10二维卷积识别任务

## Background - 背景

> 本项目为北京理工大学2022年AI专业大四小学期大作业，基于FPGA的二维卷积cifar-10识别任务。
>
> 参考项目链接：https://github.com/omarelhedaby/CNN-FPGA. 我们主要的工作是针对这个开源项目进行补充和修改，并适配到我们的任务上（如果是MNIST也没问题）。

## Introduction - 项目简介

### 技术路线

<div align=center>
    <img src=https://user-images.githubusercontent.com/60317828/187475975-771a466a-e42e-464e-8ef7-7f03745fd790.png width="700"/>
</div>

如上图所示，我们的技术路线从工作内容来说，一共分为五部分：需求定义、数据准备、模型构建、verilog编写、工程部署。

1. 在**需求定义**阶段，我们的主要任务就是确定整个小组的任务目标：针对输入为彩色图像的多对象分类任务仿真。
2. 基于我们的任务需求，在**数据准备**阶段，我们小组要做的事情就是确定要使用的数据集，并对数据集进行清洗、量化等一系列预处理工作，使数据集能为我们的模型所用。
3. 准备好数据集后就是**模型构建**阶段，这一阶段主要是针对分类任务的算法层面实现，因此我们工作的发力点在于训练一个合适的神经网络模型，并通过蒸馏、剪枝、权重量化等操作使模型转变为更适合部署在芯片上的状态。
4. 随后就是模型的**Verilog仿真**实现，在这一阶段我们的主要任务就是学习Verilog的使用，并在确定合适的开发方式后将神经网络模型仿真实现。
5. 最后一步就是在得到具体芯片的情况下，尝试将我们的仿真结果**上板烧录**，转化为可以实际应用的工程成果。

### 网络结构

下图展示了我们模型的结构，针对cifar-10数据集中的输入，进行多轮操作，得到最终的分类结果。

<div align=center>
    <img src=https://user-images.githubusercontent.com/60317828/187476004-4fc27f67-61ef-4a90-86e9-6f2627ca7e23.png width="700"/>
</div>

更具体的参数设置如下所示（激活函数使用ReLu）：

<div align=center>
    <img src=https://user-images.githubusercontent.com/60317828/187476137-4e30b148-1d4f-4205-8b35-a4103920db5c.png width="700"/>
</div>

### 识别结果

|        模型名称         |    精度    |
| :---------------------: | :--------: |
|  teacher模型（resnet）  |   0.947    |
|  student模型（lenet）   |   0.674    |
|       非蒸馏lenet       |   0.526    |
| 量化后lenet（仿真结果） | 0.55（？） |

最终的结果如上表所示，由于初学的原因，所以没有尝试使用verilog搭建更复杂的网络。因此，我们选择基于resnet模型进行蒸馏，以此来让lenet-5学到更好的结果。那么由于数据量化会损失一定精度，最终基于FPGA的CNN识别cifar-10的准确率到达55%。

## Our Work - 我们的工作

首先说明为什么参考[这个项目](https://github.com/omarelhedaby/CNN-FPGA)：

1. 源码采用层次化、模块化设计，方便阅读和理解，对小白及其友好；
2. 提供了很多写好的代码模块；
3. B站up主推荐。

然后说说我们做了哪些工作：

1. 原项目目标为手写数字识别，我们进一步拓展，研究其在cifar-10数据集上的效果；
2. 据此，我们基于pytorch设计了网络，并使用蒸馏得到更好的结果；
3. 删除第三层卷积；
4. 增加了一层全连接；
5. 在卷积层使用了relu替代tanh（自编代码）；
6. 使用maxpool替代avgpool（自编代码）；
7. 修改了源码中的一些错误（如reset命名错误，数据位宽错误等 沟通放入54）；
8. 改变了全连接层的输入维度、输出维度；
9. 编写了卷积层的testbench，并通过了仿真；
10. 自编16转32位转换器以及对应tesetbench代码；
11. 原项目独立编写了卷积和全连接，我们将二者合到了一起；
12. 上板进行测试；
13. 编写了中文注释，方便阅读。

## Module Description - 模块说明

> *更多有关其中的计算说明详见**技术报告.pdf***

### integrationConv

**说明：**

  卷积模块，对模型中的卷积部分（包含激活、池化）进行仿真，对应代码中的integrationConvPart.v。

**可配置参数：**

|    名称    |       说明       | 默认值 |
| :--------: | :--------------: | :----: |
| DATA_WIDTH |     数据位宽     |   16   |
|   ImgInW   |  输入图像的宽度  |   32   |
|   ImgInH   |  输入图像的高度  |   32   |
|  Conv1Out  |  第一层卷积输出  |   28   |
|  MvgP1out  |   最大池化输出   |   14   |
|  Conv2Out  |  第二层卷积输出  |   10   |
|  MvgP2out  |   最大池化输出   |   5    |
|   Kernel   |   卷积核的大小   |   5    |
|  DepthC1   | 第一层卷积核数量 |   6    |
|  DepthC2   | 第二层卷积核数量 |   16   |

**输入输出：**

|    名称     |  类型  |                             说明                             |                       长度                       |
| :---------: | :----: | :----------------------------------------------------------: | :----------------------------------------------: |
|  CNNinput   | input  | 输入的图像，数据从左上至右下排列，每一个像素值用半精度浮点数表示 |           ImgInW × ImgInH × DATA_WIDTH           |
|   Conv1F    | input  | 第一层卷积核权值，从第一个卷积核左上开始，到最后一个卷积核右下，每一个值用半精度浮点数表示 |      Kernel × Kernel × DepthC1× DATA_WIDTH       |
|   Conv2F    | input  | 第二层卷积核权值，从第一个卷积核左上开始，到最后一个卷积核右下，每一个值用半精度浮点数表示 | DepthC2 × Kernel × Kernel × DepthC1 × DATA_WIDTH |
| iConvOutput | output |                         输出的特征图                         |    MvgP2out × MvgP2out × DepthC2 × DATA_WIDTH    |

### Relu

**说明：**

  激活函数单元1，relu激活函数。对应代码中的UsingTheRelu16.v和activationFunction.v（二者数据位宽不同，其实有冗余，懒得改了）

**可配置参数：**

|     名称     |   说明   | 默认值 |
| :----------: | :------: | :----: |
|  DATA_WIDTH  | 数据位宽 |   16   |
| OUTPUT_NODES | 输出位宽 |   32   |

**输入输出：**（每层不同，这里仅举例）

|   名称    |  类型  |   说明   |             长度              |
| :-------: | :----: | :------: | :---------------------------: |
| input_fc  | input  | 输入特征 | Conv1Out × Conv1Out × DepthC1 |
| output_fc | output | 输出特征 | Conv1Out × Conv1Out × DepthC1 |

### MaxPoolMulti

**说明：**

  最大池化模块，可以对输入进行最大池化运算。

**可配置参数：**

|    名称    |     说明     | 默认值 |
| :--------: | :----------: | :----: |
| DATA_WIDTH |   数据位宽   |   16   |
|     D      |   数据通道   |   32   |
|     H      | 输入特征高度 |   28   |
|     W      | 输出特征宽度 |   28   |

**输入输出：**（每层不同，这里仅举例）

|   名称   |  类型  |   说明   |              长度              |
| :------: | :----: | :------: | :----------------------------: |
| apInput  | input  | 输入特征 |     H × W × D × DATA_WIDTH     |
| apOutput | output | 输出特征 | (H/2) × (W/2) × D × DATA_WIDTH |

### ANNfull

**说明：**

  全连接模块，对模型中的全连接部分（包含激活）进行仿真，对应代码中的ANNfull.v。

**可配置参数：**

|      名称      |       说明       | 默认值 |
| :------------: | :--------------: | :----: |
|   DATA_WIDTH   |     数据位宽     |   32   |
| INPUT_NODES_L1 | 第一层输入节点数 |  400   |
| INPUT_NODES_L2 | 第二层输入节点数 |  120   |
| INPUT_NODES_L3 | 第三层输入节点数 |   84   |
|  OUTPUT_NODES  |    输出节点数    |   10   |

**输入输出：**

|    名称    |  类型  |                        说明                         |            长度             |
| :--------: | :----: | :-------------------------------------------------: | :-------------------------: |
| input_ANN  | input  |         全连接层的输入，用单精度浮点数表示          | DATA_WIDTH × INPUT_NODES_L1 |
| output_ANN | output | 预测的标签值，cifar-10为10分类，需要用4位二进制表示 |              4              |

### IEEE162IEEE32

**说明：**

  精度转换模块，将16位宽浮点数转换为32位浮点数，对应代码中的IEEE162IEEE32.v。

**可配置参数：**

|     名称     |     说明     | 默认值 |
| :----------: | :----------: | :----: |
| DATA_WIDTH_1 | 输入数据位宽 |   16   |
| DATA_WIDTH_2 | 输出数据位宽 |   32   |
|    NODES     |  输出节点数  |  400   |

**输入输出：**

|   名称    |  类型  |                    说明                    |         长度         |
| :-------: | :----: | :----------------------------------------: | :------------------: |
| input_fc  | input  | 精度转换模块的输入，数据用半精度浮点数表示 | DATA_WIDTH_1 × NODES |
| output_fc | output | 精度转换模块的输出，数据用单精度浮点数表示 | DATA_WIDTH_2 × NODES |

### LeNet

**说明：**

  整个网络模块，包含两层卷积和三层全连接，对应源码中的Lenet.v.

**可配置参数：**

|     名称     |       说明       | 默认值 |
| :----------: | :--------------: | :----: |
| DATA_WIDTH_1 |  卷积层数据位宽  |   16   |
| DATA_WIDTH_2 | 全连接层数据位宽 |   32   |
|    ImgInW    |  输入图像的宽度  |   32   |
|    ImgInH    |  输入图像的高度  |   32   |
|    Kernel    |   卷积核的大小   |   5    |
|   MvgP2out   |   最大池化输出   |   5    |
|   DepthC1    | 第一层卷积核数量 |   6    |
|   DepthC2    | 第二层卷积核数量 |   16   |

**输入输出：**

|    名称     |  类型  |                             说明                             |                        长度                        |
| :---------: | :----: | :----------------------------------------------------------: | :------------------------------------------------: |
|  CNNinput   | input  | 输入的图像，数据从左上至右下排列，每一个像素值用半精度浮点数表示 |           ImgInW × ImgInH × DATA_WIDTH_1           |
|   Conv1F    | input  | 第一层卷积核权值，从第一个卷积核左上开始，到最后一个卷积核右下，每一个值用半精度浮点数表示 |      Kernel × Kernel × DepthC1× DATA_WIDTH_1       |
|   Conv2F    | input  | 第二层卷积核权值，从第一个卷积核左上开始，到最后一个卷积核右下，每一个值用半精度浮点数表示 | DepthC2 × Kernel × Kernel × DepthC1 × DATA_WIDTH_1 |
| LeNetoutput | output |                         输出的特征图                         |                         3                          |

------

## Requirements - 必要条件

- Windows
- python3.7 or up
- pytorch
- vivado

## Usage - 使用方法

1. 下载本仓库

```
mkdir CNN-FPGA
cd ./CNN-FPGA
git clone https://github.com/Robin-WZQ/CNN-FPGA.git
```

2. 训练模型

> 已提供训练好的模型。

 ```
cd ./cifar_source_torch
python main.py
 ```

3. 量化并保存

> 如果用自己训练好的模型，就将pt文件移动到quantification文件夹下。
> 
> 修改ANNfull.v和Lenet_tb.v中的权重路径，我们提供了我们模型训练好的权重文件，详见weight文件夹中。

 ```
# 权重量化
python quantification_para.py
# 输入图像量化
python quantification_img.py
 ```

4. 运行仿真

打开CNN-FPGA-Vivado，在 vivado 里，运行LeNet_tb的simulation，即可得到结果。

一开始为9，之后运行约1分钟之后变为预测的数字。

<div align=center>
    <img src=https://user-images.githubusercontent.com/60317828/187814157-ae331356-7ceb-488f-9a1f-33e08efd7c8d.png width="700"/>
</div>


## Code Tree - 文件代码树

> 文件及对应说明如下所示

|cifar_source_torch

----|distill.py # 

----|distilled_lenet5_best.pt # 

----|main.py # 

----|models.py # 

----|save_params.py # 

----|test.py # 

|CNN-FPGA-Vivado

|quantification

----|cifar-10-python # 

----|distilled_lenet5_best.pt # 

----|input_pic2_label8.txt # 

----|quantification_img.py # 

----|quantification_para.py # 

|weight

----|classifier.txt # 

----|fc1.txt # 

----|fc2.txt # 

----|layer1.txt # 

----|layer2.txt # 
