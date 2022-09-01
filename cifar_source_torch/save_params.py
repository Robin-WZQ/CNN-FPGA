from models import Lenet5, NormalCNN
import torch
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=100)

model = Lenet5()
model.load_state_dict(torch.load('./res/distilled_lenet5_best.pt'))

f = open('lenet5_parameters_cifar.txt', 'w', encoding='utf-8')
for name, params in model.named_parameters():
    print(name, params.shape)
    f.write(name + ':' + str(params.shape) + '\n')
    f.write(str(params.detach().numpy()) + '\n')
