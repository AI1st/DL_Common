import torch
from torch import nn
from d2l import torch as d2l


class Relpu_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        # 前向传播逻辑
        mask = (x > 0).float()
        temp0 = 1 + mask * x
        temp1 = torch.pow(temp0, b - 1)
        temp2 = temp1 * temp0
        result = x + a * (temp2 - 1)
        ctx.save_for_backward(a, b, mask, temp0, temp1, temp2)  # 保存输入张量，以备反向传播使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播逻辑
        a, b, mask, temp0, temp1, temp2 = ctx.saved_tensors  # 获取保存的输入张量
        grad_x = grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            # grad_x = grad_output * (1 + mask * a * b * torch.pow(1 + x, b - 1))
            grad_x = grad_output * (1 + mask * temp1 * a * b)

        if ctx.needs_input_grad[1]:
            # grad_a = grad_output * torch.pow(1 + x, b) * mask - mask
            grad_a = grad_output * (temp2 - 1)

        if ctx.needs_input_grad[2]:
            # grad_b = grad_output * a * torch.pow(1 + x, b) * torch.log(1 + x * mask) * mask
            grad_b = grad_output * a * torch.log(temp0) * temp2

        return grad_x, grad_a, grad_b


def relpu(x, a, b):
    output = Relpu_s.apply(x, a, b)
    return output


def sigmoid_distribution(s, a, b, c):
    return 1 / (1 + torch.exp(-a * (s - b))) * (1 - c) + c


class Relpu(nn.Module):
    # version1
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)
        self.weight_c = nn.Parameter(0.01 * (torch.randn(1)) + 1)  # 暂时不用

    def forward(self, x):
        mask = x > 0
        x[mask] = self.weight_c * (x[mask] + self.weight_a * torch.pow(x[mask] + 1, self.weight_b) - self.weight_a)
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()


class Relpu2(nn.Module):
    # version2
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)

    def forward(self, x):
        mask = x > 0
        x[mask] = x[mask] + self.weight_a * torch.pow(x[mask] + 1, self.weight_b) - self.weight_a
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()


class Relpu3(nn.Module):
    # version3
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)

    def forward(self, x):
        x = relpu(x, self.weight_a, self.weight_b)
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()


class Relpu4(nn.Module):
    # version4
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)

    def forward(self, x):
        x = relpu(x, self.weight_a, self.weight_b)
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()


class RelpuGlob(nn.Module):
    # version RelpuGlob
    def __init__(self, features, u_a=0, u_b=1, sigma_a=0.05, sigma_b=0.05, dim=1):
        super().__init__()
        self.features = features
        self.weight_a = nn.Parameter(sigma_a * (torch.randn(features)) + u_a)
        self.weight_b = nn.Parameter(sigma_b * (torch.randn(features)) + u_b)
        self.dim = dim

    def forward(self, x):
        view = [self.features if i == self.dim else 1 for i in range(len(x.shape))]
        x = relpu(x, self.weight_a.view(view), self.weight_b.view(view))
        return x


def init_relpus(u_a, u_b, sigma_a, sigma_b):
    def init(m):
        if type(m) == Relpu3:
            m.weight_a, m.weight_b = nn.Parameter(u_a + sigma_a * torch.randn(1)), nn.Parameter(
                u_b + torch.randn(1) * sigma_b)

    return init
