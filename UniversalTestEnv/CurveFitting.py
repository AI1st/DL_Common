import inspect
import torch
import torch.nn as nn
import torch.optim as optim
from Common.Core.Universal.NNFrameWork import NNFrameWork, cpu, gpu
from Common.Core.CustomLayers.Activations.Relpu import RelpuGlob
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch.utils.data as Data

# 生成非线性数据
x = torch.linspace(-3, 3, 100).reshape(-1, 1)
noise = torch.randn(*x.shape)
x_outer = torch.linspace(-6, 6, 1000).reshape(-1, 1)
y = 4.2 * (x + 1) ** 2 + 1.1 + noise + 1 * torch.randn(*x.shape) + 10 * torch.sin(x * 15)
y_outer = 4.2 * (x_outer + 1) ** 2 + 1.1 + 1 * torch.randn(*x_outer.shape) + 10 * torch.sin(x_outer * 15)


def get_dataset(dataset_name):
    x = torch.linspace(-3, 3, 100).reshape(-1, 1)
    if dataset_name == 'sin+poly':

        return Data.TensorDataset(x, y)
    elif dataset_name == 'piecewise':
        return Data.TensorDataset(x, y)
