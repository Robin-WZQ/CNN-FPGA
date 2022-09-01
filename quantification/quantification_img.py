import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms 
import torch

def dec2bin(x):
    x -= int(x)
    bins = []
    while x:
        x *= 2
        bins.append("1" if x>=1. else "0")
        x -= int(x)
    return "".join(bins)

def float2IEEE16(x):
    ms = "0" if x > 0 else "1"
    x = abs(x)
    x0 = int(x) # 整数部分
    x1 = x - x0 # 小数部分
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
    if len(E)>5:
        E = E[:5]
    else:
        for i in range(5 - len(E)):
            E = "0" + E
    m = m[1:]
    if len(m)>10:
        m = m[:10]
    else:
        for i in range(10 - len(m)):
            m += "0"
    y = ms+E+m
    y1 = ""
    for i in range(len(y)//4):
        y1 += hex(int(y[4*i:4*(i+1)], 2)).replace("0x", "")
    return y1

# print(halfpre2spre('0011110000000000')) # 0.1
# print(uint82float16(255))   #0101101111111000/5bf8

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507))
    ])

    filename = "./cifar-10-python/cifar-10-batches-py/test_batch"
    dataset = unpickle(filename)
    data = dataset[b"data"]
    labels = dataset[b"labels"]

    id = 2

    one_data = np.zeros((3, 1024), dtype=np.float32)
    one_data[0], one_data[1], one_data[2] = data[id][:1024], data[id][1024:2048], data[id][2048:]
    one_data = one_data.reshape((3, 32, 32))
    one_data = torch.tensor(one_data, dtype=torch.float32)
    one_data = transform(one_data)

    one_label = labels[id]
    print(f"图像的标签为 {one_label}")

    result = ""

    for i in range(1):
        for j in range(32):
            for k in range(32):
                result += float2IEEE16(one_data[i][j][k])

    f = open(f"input_pic{id}_lable{one_label}.txt", 'w', encoding="utf-8")
    f.write(result)

    img = np.reshape(data[id], (3, 32, 32))
    plt.imshow(img.transpose((1, 2, 0)))
    plt.show()