import torch
from torch.optim.optimizer import Optimizer


class TentativeSGD(Optimizer):
    """
    试探性梯度下降
    """

    def __init__(self, params, lr_start=1.0, lr_k=0.1, lr_num=3, soft_p=9):
        defaults = dict(lr_start=lr_start, lr_k=lr_k, lr_num=lr_num, soft_p=soft_p)
        super().__init__(params, defaults)
        self.lr_hist = []

    @torch.no_grad()
    def step(self, closure=None):
        """
        试探性梯度下降更新
        :param closure: 传入有关loss计算的lambda表达式
        :return: 返回单步更新后的最终loss
        """
        loss = None
        lr_effective = None
        if closure is not None:
            loss = closure()  # 初始loss获取

        for group in self.param_groups:  # self.param_groups为仅有一个元素的字典，此处相当于取出其第一个元素
            # 基本参数获取及初始化
            lr_start = group['lr_start']
            lr_k = group['lr_k']
            lr_num = group['lr_num']
            soft_p = group['soft_p']

            lrs = [lr_start * lr_k ** i for i in range(lr_num)]  # 学习率列表

            original_data = [p.data.clone() for p in group['params'] if p.grad is not None]  # 原始权重保存

            losses = [loss.item()]  # 初始化loss列表

            # 试探性学习率更新
            for lr in lrs:
                for p, od in zip(group['params'], original_data):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    p.data.add_(grad, alpha=-lr)

                if closure is not None:
                    loss = closure()
                losses.append(loss.item())

                for p, od in zip(group['params'], original_data):
                    if p.grad is None:
                        continue
                    p.data.copy_(od)

            loss_decreases = torch.tensor([losses[0] - l for l in losses[1:]])

            # # 判断loss_decreases是否全部小于0(是否学习率区间过高)
            # if torch.all(loss_decreases < 0):
            #     group['lr_start'] = group['lr_start'] * group['lr_k']
            #     return self.step(closure)
            #
            # # 判断loss_decreases是否全部大于0(是否学习率区间过低)
            # if torch.all(loss_decreases > 0) and torch.all(torch.tensor([losses]) != 0):
            #     group['lr_start'] = group['lr_start'] / group['lr_k']
            #     return self.step(closure)

            # # 计算loss_decreases_normalized
            # if torch.abs(loss_decreases).mean() < 1:
            #     epsilon = 1e-10
            #     loss_decreases_normalized = loss_decreases / (torch.abs(loss_decreases).mean() + epsilon)
            # else:
            #     loss_decreases_normalized = loss_decreases

            loss_decreases_normalized = loss_decreases

            # 学习率加权权重计算
            lr_weights = torch.softmax(loss_decreases_normalized * soft_p, dim=0)

            # 最终更新学习率计算
            lr_effective = (torch.tensor(lrs) * lr_weights).sum()

            # 最终学习率更新
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data, alpha=-lr_effective.item())

        # 学习率历史记录
        self.lr_hist.append(lr_effective.item())
        final_loss = closure()
        return final_loss


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import d2l.torch as d2l
    from torch.optim import SGD


    # 定义一个简单的神经网络
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    # 实例化网络
    net = Net()

    # 随便生成一些数据和标签来模拟训练
    data = torch.randn(5000, 784)
    target = torch.randint(0, 10, (5000,))

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 使用多尺度学习率优化器
    optimizer = TentativeSGD(net.parameters(), lr_start=1, lr_k=0.1, lr_num=3, soft_p=9)

    # 训练循环
    for epoch in range(500):
        optimizer.zero_grad()  # 清零梯度
        output = net(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        if epoch == 0:
            print(f"init_loss: {loss.item()}")
        loss = optimizer.step(lambda: criterion(net(data), target))  # 优化步骤
        if loss < 1e-16:
            break
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # print(optimizer.lr_hist)
    # print(len(optimizer.lr_hist))
    d2l.use_svg_display()
    plt.figure()
    plt.plot(optimizer.lr_hist)
    plt.show()
