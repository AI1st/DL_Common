import torch
import torch.nn as nn

def init_weights(module):
    """
    初始化网络的权重。
    对于 BN 层，将 bias 项设置为随机正态分布。
    """
    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        # 检测当前层是否为 BN 层
        module.bias.data.normal_(0, 1)  # 将 bias 项设置为随机正态分布
        module.weight.data.fill_(1)     # 将 weight 项设置为 1