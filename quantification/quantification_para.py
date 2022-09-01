from multiprocessing.spawn import import_main_path
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import ctypes


class Lenet5(nn.Module):
    '''
    所用的卷积神经网络
    '''

    def __init__(self):
        super(Lenet5, self).__init__()
        # 32, 32, 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 14, 14, 6
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 5, 5, 16
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.fc2(x)
        logits = self.classifier(x)

        return logits


def dec2bin(x):
    '''
    将十进制小数x转为对应的二进制小数
    '''
    x -= int(x)
    bins = []
    while x:
        x *= 2
        bins.append("1" if x >= 1. else "0")
        x -= int(x)
    return "".join(bins)


def float2IEEE16(x):
    '''
    float转IEEE754的半精度浮点数
    '''
    ms = "0" if x > 0 else "1"
    x = abs(x)
    x0 = int(x)  # 整数部分
    x1 = x - x0  # 小数部分
    x0 = bin(x0).replace("0b", "")
    x1 = dec2bin(x1)
    if x0[0] == "0":
        E = 15 - x1.find("1") - 1
        m = x1[x1.find("1"):]
        if E < 0:
            E = 15
            m = "00000000000"
    else:
        E = 15 + len(x0) - 1
        m = x0 + x1
    E = bin(E).replace("0b", "")
    if len(E) > 5:
        E = E[:5]
    else:
        for i in range(5 - len(E)):
            E = "0" + E
    m = m[1:]
    if len(m) > 10:
        m = m[:10]
    else:
        for i in range(10 - len(m)):
            m += "0"
    y = ms+E+m
    y1 = ""
    for i in range(len(y)//4):
        y1 += hex(int(y[4*i:4*(i+1)], 2)).replace("0x", "")
    return y1


def float2IEEE32(x):
    '''
    float转IEEE754的单精度浮点数
    '''
    ms = "0" if x > 0 else "1"
    x = abs(x)
    x0 = int(x)  # 整数部分
    x1 = x - x0  # 小数部分
    x0 = bin(x0).replace("0b", "")
    x1 = dec2bin(x1)
    if x0[0] == "0":
        E = 127 - x1.find("1") - 1
        m = x1[x1.find("1"):]
        if E < 0:
            E = 127
            m = "000000000000000000000000"
    else:
        E = 127 + len(x0) - 1
        m = x0 + x1
    E = bin(E).replace("0b", "")
    if len(E) > 8:
        E = E[:8]
    else:
        for i in range(8 - len(E)):
            E = "0" + E
    m = m[1:]
    if len(m) > 23:
        m = m[:23]
    else:
        for i in range(23 - len(m)):
            m += "0"
    y = ms+E+m
    y1 = ""
    for i in range(len(y)//4):
        y1 += hex(int(y[4*i:4*(i+1)], 2)).replace("0x", "")
    return y1


if __name__ == '__main__':
    model = Lenet5()
    model.load_state_dict(torch.load("distilled_lenet5_best.pt"))
    for name in model.state_dict():
        # print(name, '\t', model.state_dict()[name].shape)
        # print(model.state_dict()[name])

        # 卷积层权重量化
        if name in ["layer1.0.weight", "layer2.0.weight"]:
            fname = name.split(".")[0]
            Tensor = model.state_dict()[name]
            s1, s2, s3, s4 = Tensor.shape
            with open("parameters/"+fname+".txt", "w", encoding="utf-8") as f:
                for i in range(s1):
                    for j in range(s2):
                        for k in range(s3):
                            for t in range(s4):
                                f.write(float2IEEE16(Tensor[i][j][k][t]))
                    f.write("\n")
        # 全连接层权重量化
        if name in ["fc1.0.weight", "fc2.0.weight", "classifier.weight"]:
            fname = name.split(".")[0]
            Matrix = model.state_dict()[name].T
            with open("parameters/"+fname+".txt", "w", encoding="utf-8") as f:
                for i in range(Matrix.shape[0]):
                    for j in range(Matrix.shape[1]):
                        f.write(float2IEEE32(Matrix[i][j])+"\n")
