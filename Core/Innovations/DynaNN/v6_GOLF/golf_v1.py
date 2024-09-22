from d2l import torch as d2l
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
import numpy as np


def get_oscillation_index(grad, m, m_prime, oscillation_index_p, beta=0.5, epsilon=1e-6):
    """
    更新振荡指标
    :param grad: 传入参数的梯度
    :param m: 传入原来的参数状态参数m
    :param m_prime: 传入原来的状态参数m_prime
    :param oscillation_index_p: 传入原来的状态指标oscillation_index_p
    :param beta: 设定的泄露动量更新率
    :param epsilon: 一个小值，防止数值问题造成的计算失误
    :return:
    """
    with torch.no_grad():
        abs_grad = torch.abs(grad)
        mask = abs_grad > epsilon
        # 确保更新是在原地进行的
        if mask.any():
            m.masked_scatter_(mask, m[mask] * beta + grad[mask] * (1 - beta))
            m_prime.masked_scatter_(mask, m_prime[mask] * beta + abs_grad[mask] * (1 - beta))
        oscillation_index_raw = (m + epsilon * 1e-16) / (m_prime + epsilon * 1e-16)
        oscillation_index_abs = torch.abs(oscillation_index_raw)
        if oscillation_index_p is not None:
            oscillation_index_filtered = (oscillation_index_p + oscillation_index_raw) / 2
            oscillation_index_abs_filtered = (torch.abs(oscillation_index_p) + oscillation_index_abs) / 2
        else:
            oscillation_index_filtered = (oscillation_index_raw + torch.sign(oscillation_index_raw)) / 2
            oscillation_index_abs_filtered = (oscillation_index_abs + 1) / 2
        oscillation_index_f = 2 * torch.abs(oscillation_index_filtered) - oscillation_index_abs_filtered
    return oscillation_index_raw, oscillation_index_abs, oscillation_index_filtered, oscillation_index_abs_filtered, oscillation_index_f


def calculate_increase_lr0(od):
    mask0 = (od <= 0.9).float()
    mask1 = (od > 0.9).float()
    return (0.415 * od + 0.670) * mask0 + (2.0 * od - 0.7565) * mask1


def calculate_increase_lr(od):
    mask0 = (od <= 0.9).float()
    mask1 = (od > 0.9).float()
    return (0.415 * od + 0.670) * mask0 + (2.0 * od - 0.7565) * mask1


class GOLF(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.5, epsilon=1e-6, weight_decay=0):
        # 设置默认参数
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.history = {}  # 存储振荡指标的历史记录

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # 检查是否为稀疏梯度
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GOLF does not support sparse gradients')

                state = self.state[param]

                # 初始化状态
                if len(state) == 0:
                    state['M'] = torch.zeros_like(param.data)
                    state['M_prime'] = torch.zeros_like(param.data)
                    state['oscillation_index'] = None
                    # 初始化各参数的学习率
                    state['lr'] = group['lr'] * torch.ones_like(param.data)

                # state
                M, M_prime = state['M'], state['M_prime']
                oscillation_index_p = state['oscillation_index']
                # group
                beta = group['beta']
                epsilon = group['epsilon']

                # 更新振荡指标并记录
                oscillation_index_raw, oscillation_index_abs, oscillation_index_filtered, oscillation_index_abs_filtered, oscillation_index_f = get_oscillation_index(
                    grad, M, M_prime, oscillation_index_p, beta, epsilon)
                state['oscillation_index'] = oscillation_index_raw

                # 参数更新(使用学习率、权重衰减等)
                decrease = calculate_increase_lr(oscillation_index_f)
                state['lr'] *= decrease
                step_size = state['lr']
                if group['weight_decay'] != 0:
                    grad = grad.add(param.data, alpha=group['weight_decay'])

                # param.data.add_(grad, alpha=-step_size)
                param.data = param.data - step_size * grad
                # print(step_size)

                # 存储振荡指标到历史记录字典中
                param_id = id(param)
                if param_id not in self.history:
                    self.history[param_id] = []
                self.history[param_id].append({
                    'oscillation_index_raw': oscillation_index_raw.clone(),
                    'oscillation_index_abs': oscillation_index_abs.clone(),
                    'oscillation_index_filtered': oscillation_index_filtered.clone(),
                    'oscillation_index_abs_filtered': oscillation_index_abs_filtered.clone(),
                    'oscillation_index_c': (oscillation_index_abs.clone() - 2 * (
                            oscillation_index_abs.clone() - torch.abs(oscillation_index_filtered.clone()))),
                    'oscillation_index_c2': oscillation_index_f
                })

        return loss
