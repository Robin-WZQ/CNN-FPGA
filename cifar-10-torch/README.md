# 文件说明

**cifar-source：**源代码文件夹

- cifar-10-python：数据集文件夹
- res：模型存储文件夹
  - distilled_lenet5_best.pt：蒸馏得到的Lenet5模型参数文件
- train.py：Teacher model训练代码
- distill.py：模型蒸馏训练代码
- test.py：验证评估代码
- save_params.py：参数导出代码
- models.py：模型结构代码
- Lenet5_parameters_cifar.txt：导出模型参数文件

# 实验环境

Python==3.9

torch==1.9.0+cu111

torchvision==0.10.0+cu111

numpy==1.20.3

# 代码说明

![未命名绘图.drawio](C:\Users\Caesar\Desktop\figures\1\未命名绘图.drawio.png)