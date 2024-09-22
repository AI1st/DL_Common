from torch import nn
from Common.Core.Universal.NNFrameWork import NNFrameWork


class Sequential(nn.Sequential, NNFrameWork):  # 使得sequential架构具有NNFrameWork的性质
    def forward(self, input_x):  # 重写forward方法以避免多重继承时forward方法无法被元类正确修饰
        return super().forward(input_x)


class Linear(nn.Linear, NNFrameWork):
    def forward(self, input_x):  # 重写forward方法以避免多重继承时forward方法无法被被元类正确修饰
        return super().forward(input_x)
