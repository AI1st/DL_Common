import torch
import torch.nn as nn


class LaplaceActivation(nn.Module):
    """
    Laplace激活函数(sin指数衰减激活函数，可指定bias)
    """

    def __init__(self, features, bias=True):
        """
        Initialize the LaplaceActivation layer.

        Args:
            features (int): The number of input features.
            bias (bool, optional): If True, adds a learnable bias to the activation function. Defaults to True.

        Attributes:
            bias (bool): Whether to add a learnable bias to the activation function.
            a (nn.Parameter): A learnable parameter representing the scaling factor for the input.
            b (nn.Parameter, optional): A learnable bias parameter, only added if bias is True.
        """
        super().__init__()
        self.bias = bias
        self.a = nn.Parameter(torch.ones(features))
        if bias:
            self.b = nn.Parameter(torch.randn(features))

    def forward(self, x):
        view_shape = [1] * x.dim()
        view_shape[1] = -1

        a = self.a.view(view_shape)
        if self.bias:
            b = self.b.view(view_shape)
            return torch.exp(-torch.abs(a * x + b)) * torch.sin(x)

        return torch.exp(-torch.abs(a * x)) * torch.sin(x)
