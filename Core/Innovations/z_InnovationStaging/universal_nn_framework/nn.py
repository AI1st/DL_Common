import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

"""
模型快速训练四部曲：
1.模型定义(及初始化)
2.优化器定义
3.损失函数定义
4.模型训练
"""


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')


def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')


def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


class NN(nn.Module):
    def __init__(self, model=None, optimizer=None, criterion=None, device=try_gpu(), init_f=None):
        super().__init__()
        ################################
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip_value = None
        if init_f is not None and model is not None:
            self.model.apply(init_f)
        #################################
        self.iter_times = 0  # input_num
        self.loss_history = []
        self.temp_loss_history = []
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def train_real_time(self, x, target):
        self.train()
        # 更新队列及迭代次数
        self.iter_times += 1

        # 学习及更新
        loss = self._get_gradient(x, target)
        self._self_update()

        # 学习历史记录添加
        self.loss_history.append(loss)

    def train_fixed(self, x, target, epochs_end_flag=False):
        self.train()

        # 学习及更新
        loss = self._get_gradient(x, target)
        self._self_update()

        if not epochs_end_flag:
            self.temp_loss_history.append(loss)
        if epochs_end_flag:
            self.temp_loss_history.append(loss)
            self.loss_history.append(np.mean(self.temp_loss_history))
            self.temp_loss_history = []
            # print(self.loss_history)
            # 更新队列及迭代次数
            self.iter_times += 1

    def _get_gradient(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        # 学习及更新
        output = self(x)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        return loss.item()

    def _self_update(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)  # 梯度截断
        self.optimizer.step()

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, lr=0.01, momentum=0.0, clip_value=float("inf"), optimizer="default"):
        self.clip_value = clip_value
        if optimizer == "default":
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_device(self, device):
        self.device = device
        self.to(self.device)

    def predict(self, x):
        self.eval()
        return self(x)

    def save_model(self, name="model.pt"):
        torch.save(self.state_dict(), name)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


def train_nn(model, data_loader, epochs, save=False, plot_history=False, device=try_gpu()):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(data_loader):
            if i == len(data_loader) - 1:
                flag = True
            else:
                flag = False
            x = x.to(device)
            y = y.to(device)
            model.train_fixed(x, y, epochs_end_flag=flag)
    if save:
        model.save_model("model.pt")
    if plot_history:
        plt.figure()
        backend_inline.set_matplotlib_formats('svg')
        plt.plot(model.loss_history)
        plt.xlabel("loss")
        plt.ylabel("epochs")
        plt.show()
    return model.loss_history[-1]
