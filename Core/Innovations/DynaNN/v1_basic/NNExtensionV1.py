import torch
from torch import nn
from torch import optim
from Common.function.global_gcc import RelpuGlob


class Relpu1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relpu = RelpuGlob(1)

    def forward(self, x):
        return self.relpu(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size, use_bn=False, activation=Relpu1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(input_size))  # first bn
        if use_bn:
            self.layers.append(nn.Linear(input_size, hidden_size, bias=False))  # first layer & activation
            self.layers.append(nn.BatchNorm1d(hidden_size))
        else:
            self.layers.append(nn.Linear(input_size, hidden_size, bias=True))
        self.layers.append(activation())
        for _ in range(depth):
            if use_bn:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.BatchNorm1d(hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_size, output_size))


class DataNorm(nn.Module):
    """
    既可以用作输入规范化，也可以用作输出规范化
    """
    def __init__(self):
        super().__init__()
        self.mean = 0
        self.mean_square = 0
        self.iter_times = 0
        self.var = 0

    def forward(self, x):
        self.iter_times += 1
        alpha = 1 / self.iter_times
        self.mean = self.mean + alpha * (x.mean(dim=0) - self.mean)
        self.mean_square = self.mean_square + alpha * ((x ** 2).mean(dim=0) - self.mean_square)
        self.var = self.mean_square - self.mean ** 2
        return (x - self.mean) / self.var ** 0.5

    def out_amplify(self, y):
        return y * self.var ** 0.5 + self.mean
