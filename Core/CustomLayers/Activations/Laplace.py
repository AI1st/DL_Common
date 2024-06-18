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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # 设置不同的参数 (a, b) 并绘制对应的曲线
    def plot_laplace_activation(parameters):
        x = np.linspace(-10, 10, 400)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

        for idx, (a_value, b_value) in enumerate(parameters):
            model = LaplaceActivation(features=1, bias=(b_value is not None))
            model.a = nn.Parameter(torch.tensor([a_value], dtype=torch.float32))
            if b_value is not None:
                model.b = nn.Parameter(torch.tensor([b_value], dtype=torch.float32))

            with torch.no_grad():  # 禁用梯度计算
                y_tensor = model(x_tensor)
            y = y_tensor.numpy().flatten()

            label = f'a={a_value}, b={b_value if b_value is not None else 0}'
            plt.plot(x, y, label=label)

        plt.xlabel('x')
        plt.ylabel('Activation Value')
        plt.title('Laplace Activation Function')
        plt.legend()
        plt.grid(True)
        plt.show()


    # 不同的参数设置
    params = [
        (1, 0),
        (1, 1),
        (1.5, 0.5),
        (2, -0.5),
        (0.5, -1)
    ]

    plot_laplace_activation(params)
