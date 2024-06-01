import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from Common.function.machine_learning import plot
from Common.function_version_iteration.TentativeSGD.TSGD import TentativeSGD


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


class NNFrameWork(nn.Module):
    """
    通用神经网络训练框架
    核心框架结构梳理：

    >训练类方法:
    train_real_time
    train_fixed
        >训练函数接口(interface):
        _train_fixed_implementation
        >训练核心组件:
        _check_feasibility
        _get_gradient
        _self_update

    >模型衡量/应用方法:
    predict
    evaluate_loss

    >模型信息输入输出:
    plot_hist
    save_model
    load_model
    load_optimizer

    >模型基本架构设置:
    set_optimizer
    set_optimizer_tsgd
    set_weights_init
    set_criterion
    set_device
    """

    def __init__(self):
        super().__init__()
        ################################
        self.optimizer = None
        self.criterion = None
        self.device = None
        self.clip_value = None
        self.init_fs = None
        #################################
        self.iter_times = 0  # input_num
        self.loss_history = []
        self.temp_loss_history = []
        self.to(self.device)

    def train_real_time(self, x, target):
        self._check_feasibility()
        self.train()

        # 更新队列及迭代次数
        self.iter_times += 1

        # 学习及更新
        loss = self._get_gradient(x, target)
        self._self_update()

        # 学习历史记录添加
        self.loss_history.append(loss)

    def train_fixed(self, data_iter, epochs, save_path=None, plot_hist=True,
                    other_options=None):
        self._check_feasibility()
        self.train()
        loss = self._train_fixed_implementation(data_iter, epochs, other_options)
        ###############################save and plot hist##############################
        if save_path is not None:
            torch.save(self.state_dict(), save_path + ".pt")
            torch.save(self.optimizer.state_dict(), save_path + '_optimizer_state.pth')
            with open(save_path + '_train_hist.pkl', 'wb') as f:
                pickle.dump(self.loss_history, f)
        if plot_hist:
            self.plot_hist()
        ###############################save and plot hist##############################
        return loss

    def _train_fixed_implementation(self, data_iter, epochs, other_options=None):

        loss_threshold = None  # loss阈值，用于判断迭代退出条件
        if other_options is not None:
            loss_threshold = other_options['loss_threshold']

        for epoch in range(epochs):
            for i, (x, y) in enumerate(data_iter):
                x = x.to(self.device)
                y = y.to(self.device)
                ##################training################
                # 学习及更新
                loss = self._get_gradient(x, y)
                if loss_threshold is not None:  # 判断loss是否已经达到要求
                    if loss < loss_threshold:
                        return loss
                self._self_update(x, y)
                # 学习历史记录添加
                if i == len(data_iter) - 1:
                    self.temp_loss_history.append(loss)
                    self.loss_history.append(np.mean(self.temp_loss_history))
                    self.temp_loss_history = []
                    # 更新队列及迭代次数
                    self.iter_times += 1
                else:
                    self.temp_loss_history.append(loss)
                ##################training################
        return self.loss_history[-1]

    def _check_feasibility(self):
        assert self.optimizer is not None, "No optimizer specified!"
        assert self.criterion is not None, "No criterion specified!"
        assert self.device is not None, "No device specified!"

    def _get_gradient(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        # 学习及更新
        output = self(x)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        return loss.item()

    def _self_update(self, data=None, target=None):
        if self.clip_value is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)  # 梯度截断
        self.optimizer.step()

    def predict(self, x):
        self.eval()
        return self(x)

    def evaluate_loss(self, data_iter):
        loss_list = []
        for i, (x, y) in enumerate(data_iter):
            x = x.to(self.device)
            y = y.to(self.device)
            ##################evaluate################
            with torch.no_grad():
                # 计算loss
                output = self.predict(x)
                loss = self.criterion(output, y)
                loss_list.append(loss.cpu())
            ##################evaluate################
        return np.mean(loss_list)

    def plot_hist(self):
        plt.figure()
        backend_inline.set_matplotlib_formats('svg')
        plt.plot(self.loss_history)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

    def save_model(self, model_name="model"):
        torch.save(self.state_dict(), model_name + ".pt")
        torch.save(self.optimizer.state_dict(), model_name + '_optimizer_state.pth')
        with open(model_name + '_train_hist.pkl', 'wb') as f:
            pickle.dump(self.loss_history, f)

    def load_model(self, model_path_without_suffix):
        self.load_state_dict(torch.load(model_path_without_suffix + ".pt"))
        with open(model_path_without_suffix + '_train_hist.pkl', 'rb') as f:
            self.loss_history = pickle.load(f)

    def load_optimizer(self, model_path_without_suffix):
        self.optimizer.load_state_dict(torch.load(model_path_without_suffix + '_optimizer_state.pth'))

    def set_optimizer(self, lr=0.01, momentum=0.0, clip_value=float("inf"), optimizer="default"):
        self.clip_value = clip_value
        if optimizer == "default":
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            self.optimizer = optimizer
        print(self.optimizer)

    def set_weights_init(self, init_f):
        self.init_fs.append(init_f)
        self.apply(init_f)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_device(self, device):
        self.device = device
        self.to(self.device)
