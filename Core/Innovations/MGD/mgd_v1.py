import torch
import torch.nn as nn
import torch.optim as optim


class MGDBoundary(optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 alpha=0.99,
                 beta=5e5,
                 boundary=2,
                 lr_max=10,
                 lr_min=1e-5,
                 weight_decay=0
                 ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0.0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError('Invalid alpha value: {}'.format(alpha))
        if beta < 0.0:
            raise ValueError('Invalid beta value: {}'.format(beta))

        defaults = dict(
            lr=lr,
            alpha=alpha,
            beta=beta,
            boundary=boundary,
            lr_max=lr_max,
            lr_min=lr_min,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        第一次参数更新: 仅保留梯度数据，不更改学习率
        :param closure:
        :return:
        """
        loss = None
        if closure is not None:
            loss = closure()  # 第一次前向传播与反向传播

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'MGD does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                # 初始化历史梯度信息(每次调用次方法时，都将获得初始化的历史梯度，以避免小批量时的冲突问题)
                if len(state) == 0:
                    state['grad_hist'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ), 0
                    )
                    state['lr'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ), group['lr']
                    )

                # 第一次状态参数更新
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state['grad_hist'] = grad

                # 执行梯度下降
                # 计算权重变化量
                weight_change = -state['lr'] * grad
                # 计算截断阈值
                threshold = group['boundary'] * torch.abs(p.data)
                # 使用 clamp 函数进行截断
                clamped_weight_change = torch.clamp(weight_change, -threshold, threshold)
                # 权重更新
                p.data.add_(clamped_weight_change)

        return loss

    def step_follow(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()  # 第二次前向传播与反向传播

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'MGD does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                if len(state) == 0:  # 如果没有初始化变量则先初始化
                    state['grad_hist'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ), 0
                    )
                    state['lr'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ), group['lr']
                    )

                if group['weight_decay'] != 0:  # 计算考虑了weight_decay后的梯度
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # 计算学习率的梯度
                lr_grad = - state['grad_hist'] * grad

                # 更新历史梯度记录
                state['grad_hist'] = grad

                # 更新学习率
                alpha = group['alpha']
                beta = group['beta']
                lr_gain = 1 - alpha * torch.tanh(beta * lr_grad)
                state['lr'].data = state['lr'].data * lr_gain
                state['lr'].data = torch.clamp(state['lr'].data, min=group['lr_min'], max=group['lr_max'])

                # 执行梯度下降
                # 计算权重变化量
                weight_change = -state['lr'] * grad
                # 计算截断阈值
                threshold = group['boundary'] * torch.abs(p.data)
                # 使用 clamp 函数进行截断
                clamped_weight_change = torch.clamp(weight_change, -threshold, threshold)
                # 权重更新
                p.data.add_(clamped_weight_change)

        return loss
