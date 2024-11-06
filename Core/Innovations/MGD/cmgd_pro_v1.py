import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CMGDPro(optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1,
                 alpha_reduce=0.99,
                 alpha_increase=0.99,
                 beta=2e2,
                 boundary=0.5,
                 lr_max=10,
                 lr_min=1e-5,
                 weight_decay=0
                 ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0.0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        if alpha_reduce < 0.0 or alpha_reduce > 1.0:
            raise ValueError('Invalid alpha value: {}'.format(alpha_reduce))
        if alpha_increase < 0.0:
            raise ValueError('Invalid alpha value: {}'.format(alpha_increase))
        if beta < 0.0:
            raise ValueError('Invalid beta value: {}'.format(beta))
        if lr_min < 0 or lr_max < 0:
            raise ValueError('Invalid learning rate range: {}-{}'.format(lr_min, lr_max))
        if lr_min > lr_max:
            raise ValueError('Invalid learning rate order: {}>{}'.format(lr_min, lr_max))
        if boundary <= 0:
            raise ValueError('Invalid boundary value: {}'.format(boundary))

        defaults = dict(
            lr=lr,
            alpha_reduce=alpha_reduce,
            alpha_increase=alpha_increase,
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
                # # 计算截断阈值增益
                # clamped_gain = torch.abs(p.data).mean() / torch.abs(p.data)
                # 计算截断阈值
                threshold = group['boundary'] * torch.abs(p.data).mean()
                # 使用 clamp 函数进行截断
                clamped_weight_change = torch.clamp(weight_change, -threshold, threshold)
                # print(f"p:{p.data}")
                # print(f"grad:{grad}")
                # print(f"clamped_weight_change:{clamped_weight_change}")
                # 权重更新
                p.data.add_(clamped_weight_change)
                # 更新学习率
                mask = (clamped_weight_change != weight_change)  # mask中true的部分代表被截断的部分
                new_lr = torch.where(mask, torch.abs(clamped_weight_change / grad), state['lr'])  # 当mask对应部分为true(被截断)，则使用调整后的学习率，否则使用原学习率
                state['lr'] = new_lr

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
                lr_grad = torch.sqrt(torch.abs(lr_grad)) * torch.sign(lr_grad)
                # print(lr_grad)
                # # 拉伸学习率的梯度
                # epsilon = 1e-20
                # scale_rate = 1 / (torch.max(lr_grad) - torch.min(lr_grad) + epsilon)
                # lr_grad *= scale_rate
                # print(f"lr_grad:{lr_grad}")
                # print(f"lr_grad_sqrt:{torch.abs(lr_grad)**0.5}")
                # print(f"lr_grad_ln:{torch.log(torch.abs(lr_grad))}")

                # 更新历史梯度记录
                state['grad_hist'] = grad

                # 更新学习率
                alpha_reduce = group['alpha_reduce']
                alpha_increase = group['alpha_increase']
                beta = group['beta']
                compromise = torch.tanh(beta * lr_grad)
                lr_gain = 1 - (alpha_reduce * (compromise > 0).float() + alpha_increase * (
                        compromise <= 0).float()) * compromise
                state['lr'].data = state['lr'].data * lr_gain
                state['lr'].data = torch.clamp(state['lr'].data, min=group['lr_min'], max=group['lr_max'])

                # 执行梯度下降
                # 计算权重变化量
                weight_change = -state['lr'] * grad
                # 计算截断阈值
                threshold = group['boundary'] * torch.abs(p.data).mean()
                # 使用 clamp 函数进行截断
                clamped_weight_change = torch.clamp(weight_change, -threshold, threshold)
                # print(f"p:{p.data}")
                # print(f"grad:{grad}")
                # print(f"clamped_weight_change:{clamped_weight_change}")
                # 权重更新
                p.data.add_(clamped_weight_change)
                # 更新学习率
                mask = (clamped_weight_change != weight_change)  # mask中true的部分代表被截断的部分
                new_lr = torch.where(mask, torch.abs(clamped_weight_change / grad), state['lr'])  # 当mask对应部分为true(被截断)，则使用调整后的学习率，否则使用原学习率
                state['lr'] = new_lr

        return loss