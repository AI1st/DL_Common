import os
import torch
from torch import nn


class ShowLayerOutputStatics(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        os.makedirs("layer_data_output", exist_ok=True)
        with open(f'layer{self.n}.txt', 'w') as f:
            f.write("")

    def forward(self, x):
        with open(f'layer{self.n}.txt', 'a') as f:
            f.write(str(x.detach().mean()) + "\n")
        return x


class ShowLayerOutputStatics2(nn.Module):

    def __init__(self):
        super().__init__()
        os.makedirs("layer_data_output", exist_ok=True)
        folder_path = './layer_data_output/'
        file_list = os.listdir(folder_path)
        self.length = len(file_list)
        with open(f'./layer_data_output/layer{self.length + 1}.txt', 'w') as f:
            f.write("")

    def forward(self, x):
        with open(f'./layer_data_output/layer{self.length + 1}.txt', 'a') as f:
            f.write(str(x.detach().mean()) + "\n")
        return x
