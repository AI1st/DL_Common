import torch
import torch.nn as nn
import torch.optim as optim


class BioSGD(optim.Optimizer):
    def __init__(self, params, delta_W_max, beta, deactivation_rate):
        defaults = dict(delta_W_max=delta_W_max, beta=beta, deactivation_rate=deactivation_rate)
        super().__init__(params, defaults)
        self.delta_w_sum = [torch.zeros_like(p) for p in self.param_groups[0]['params']]
        self.alpha_hist = []
        self.params = params

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        with torch.no_grad():
            grads = [None] * len(self.delta_w_sum)

            for i, group in enumerate(self.param_groups):
                for j, param in enumerate(group['params']):
                    if param.grad is None:
                        continue
                    grads[j] = param.grad

            # 计算全局缩放比例
            gamma = group['delta_W_max'] / max(grad.abs().max().item() for grad in grads if grad is not None)

            for i, group in enumerate(self.param_groups):
                alphas = []
                for j, param in enumerate(group['params']):
                    if param.grad is None:
                        continue
                    delta_w_update = gamma * grads[j]
                    self.delta_w_sum[j] += delta_w_update.abs()

                    # 计算神经元活性系数
                    alpha = 1 / (1 + group['beta'] * self.delta_w_sum[j])
                    # 神经元失活判定
                    alpha = alpha * (alpha > group['deactivation_rate']).float()
                    alphas.append(alpha)  # 增加alpha到记录

                    param.add_(-alpha * delta_w_update)

                self.alpha_hist.append(alphas)  # 增加alphas到记录
        return loss

    def reset_hy_params(self, delta_W_max, beta, deactivation_rate):
        defaults = dict(delta_W_max=delta_W_max, beta=beta, deactivation_rate=deactivation_rate)
        super().__init__(self.params, defaults)
        # 重新加载历史数据
        self.delta_w_sum = [torch.zeros_like(p) for p in self.param_groups[0]['params']]
        self.alpha_hist = []


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from Common.Core.CustomLayers.Activations.Laplace import LaplaceActivation

    # 数据生成
    n_samples = 100
    X = torch.linspace(-3, 3, n_samples).reshape(n_samples, 1)
    y = (0.5 * X ** 2 + 2 + 0.1 * torch.randn(n_samples, 1) - 0.1 * X ** 3) * 0.1 + torch.sin(3 * X)

    # 线性回归模型定义
    net = nn.Sequential(
        nn.Linear(1, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 5, bias=False),
        nn.BatchNorm1d(5),
        LaplaceActivation(5),
        nn.Linear(5, 1)
    )

    model = net
    criterion = nn.MSELoss()

    # 初始化优化器
    delta_W_max = 0.5
    beta = 1
    deactivation_rate = 0.01
    optimizer = BioSGD(model.parameters(), delta_W_max, beta, deactivation_rate)

    # 训练模型
    num_epochs = 500
    all_losses = []

    for epoch in range(num_epochs):
        # 正向传播
        optimizer.zero_grad()
        y_pred = model(X)

        # 计算损失
        loss = criterion(y_pred, y)
        all_losses.append(loss.item())

        # 反向传播计算梯度
        loss.backward()

        # 使用自定义优化器更新参数
        optimizer.step()

    # 可视化结果
    plt.plot(all_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()

    model.eval()
    # 可视化拟合结果
    pred_y = model(X).detach().numpy()
    plt.scatter(X.numpy(), y.numpy(), color='blue', label='Real Data')
    plt.plot(X.numpy(), pred_y, color='red', label='Fitted Line')
    plt.legend()
    plt.show()
    print(optimizer.delta_w_sum)
    print(optimizer.alpha_hist[-1])
