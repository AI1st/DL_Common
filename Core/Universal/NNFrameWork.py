# 基于torch 2.1.1
import torch
from torch import nn
from torch import optim
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


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


class ForwardSwitchMeta(type):
    """
    定义前向传播预测、带梯度训练模式切换元类
    """

    def __new__(cls, name, bases, attrs):
        def predict_switch(func):
            def wrapper(self, *args, **kwargs):
                if hasattr(self, 'predict_mode') and self.predict_mode:
                    # print("predict_mode")
                    self.eval()
                    with torch.no_grad():
                        return func(self, *args, **kwargs)
                return func(self, *args, **kwargs)

            return wrapper

        # 获取类的属性
        forward = attrs.get('forward', None)
        if forward is not None:
            # 如果类定义了 forward 方法，则应用装饰器
            attrs['forward'] = predict_switch(forward)
        return super().__new__(cls, name, bases, attrs)


def get_params(model, lr_predecessor=None):
    params_list = []  # 参数与学习率列表
    status = False  # 指示在该模型的子模块内是否存在其它学习率的设置
    update_mark = True  # 指示模型是否被添加到优化器
    lr = lr_predecessor  # 前继节点的设置学习率

    if hasattr(model, 'update_mark') and not model.update_mark:  # 判断该节点是否需要更新参数
        update_mark = False
        return [], status, update_mark

    if hasattr(model, 'lr') and model.lr is not None:  # 判断该节点是否存在学习率设置
        status = True
        lr = model.lr

    for child in model.children():
        params_list_child, status_child, update_mark_child = get_params(child, lr)
        if not update_mark_child:  # 判断子节点是否需要更新参数，如果不需要，则跳过本回合
            continue
        if not params_list_child:  # 判断child是否为最底层的节点(当child没有子节点时，其不进入循环，返回列表为空)
            if status_child:  # 判断底层节点是否有自定义的学习率
                params_list_child = [{'params': child.parameters(), 'lr': child.lr}]
            elif lr is not None:  # 判断父节点是否有传递的学习率
                params_list_child = [{'params': child.parameters(), 'lr': lr}]
            else:  # 无学习率要求
                params_list_child = [{'params': child.parameters()}]
        if status_child:  # 只要子网络组分中存在一个模型有学习率设置，则需要拆分网络模块表以添加该学习率
            status = True
        params_list.extend(params_list_child)

    if not status and lr is None and params_list:  # 表明该模型中没有找到不同的学习率组件且没有指定的前继学习率
        # 同时，需要确保模型中的params_list非空(如果为空，则为所有子节点都被标记为不更新)
        return [{'params': model.parameters()}], status, update_mark

    return params_list, status, update_mark


class NNFrameWork(nn.Module, metaclass=ForwardSwitchMeta):
    """
    通用神经网络训练框架
    核心框架结构梳理：
        >模型内部参数设置规则：
            1. 设置模块的学习率：self.lr=<设置的学习率>
            2. 标记模块是否需要更新参数：self.update_mark = True/False
            3. 设置前向传播按照训练模式还是推理模式(不计算参数梯度，且使用eval模式)：self.predict_mode = True/False

        >训练类方法:
        train_real_time(x, target) -> loss
        train_fixed(data_iter, epochs, [save_path, plot_hist, other_options]) -> loss <from last time>
            >训练函数接口(interface):
            _train_fixed_implementation
            >训练核心组件:
            _check_feasibility
            _get_gradient
            _self_update

        >模型衡量/应用方法:
        predict(with gradient/without gradient)
        evaluate_loss
        evaluate_accuracy

        >模型信息输入输出:
        plot_hist
        save_model
        load_model
        load_optimizer

        >模型基本架构设置:
        set_optimizer
            >获取送入optimizer携带学习率的权重组:
            get_param_groups
        set_weights_init
        set_criterion
        set_device
    """

    def __init__(self):
        super().__init__()
        ################################
        self.optimizer = None
        self.criterion = None
        self.clip_value = None
        self.init_fs = None
        self.device = None
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

    def train_fixed(self, data_iter, epochs, save_path=None, plot_hist=True, show_progress=True,
                    other_options=None):
        self._check_feasibility()
        self.train()
        loss = self._train_fixed_implementation(data_iter, epochs, show_progress, other_options)
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

    def _train_fixed_implementation(self, data_iter, epochs, show_progress=True, other_options=None):
        avg_loss = None  # avg_loss初始化
        loss_threshold = None  # loss阈值，用于判断迭代退出条件
        if other_options is not None:
            loss_threshold = other_options['loss_threshold']
        total_length = len(data_iter)  # 获取迭代器的长度(以避免tqdm超界)

        for epoch in range(epochs):
            if show_progress:
                data_iter = tqdm(data_iter,
                                 desc=f'Epoch {epoch + 1}/{epochs}',
                                 leave=False,
                                 total=total_length)
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
                ##################training################
                ##################history append################
                # epoch内学习历史记录添加
                self.temp_loss_history.append(loss)
                # 更新进度条的后缀信息，显示当前的平均loss
                if show_progress:
                    avg_loss = np.mean(self.temp_loss_history)
                    data_iter.set_postfix(loss=f'{avg_loss:.4f}')
                # epoch的平均学习记录添加
                if i == total_length - 1:  # 使用从原始迭代器得到的长度，防止程序递归调用tqdm修改后的迭代器函数__len__
                    if not show_progress:
                        avg_loss = np.mean(self.temp_loss_history)
                    self.loss_history.append(avg_loss)
                    self.temp_loss_history = []
                    # 更新队列及迭代次数
                    self.iter_times += 1
                ##################history append################
        return self.loss_history[-1]

    def _check_feasibility(self):
        assert self.optimizer is not None, "No optimizer specified! Use your_model.set_optimizer(lr=your_lr) instead!"
        assert self.criterion is not None, "No criterion specified! Use your_model.set_criterion(Criterion()) instead!"
        assert self.device is not None, "No device specified! Use your_model.set_device() instead!"

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

    def predict(self, x, no_grad=False):
        self.eval()
        if no_grad:
            with torch.no_grad():
                return self(x)
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

    def evaluate_accuracy(self, data_iter):
        self.eval()  # 将模型设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 在评估模式下不需要计算梯度
            for data in data_iter:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # 移动数据到设备（如GPU）
                outputs = self.predict(inputs, no_grad=True)
                _, predicted = torch.max(outputs, 1)  # 获取每行最大值的位置（即预测的类别）
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

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

    def set_optimizer(self, lr=0.01, momentum=0.0, weight_decay=0, dampening=0, nesterov=False,
                      clip_value=float("inf"), optimizer="default"):
        self.clip_value = clip_value
        if optimizer == "default":
            self.optimizer = optim.SGD(self.get_param_groups(), lr=lr,
                                       momentum=momentum, weight_decay=weight_decay,
                                       dampening=dampening, nesterov=nesterov)
        else:
            self.optimizer = optimizer
        print(self.optimizer)

    def get_param_groups(self):
        """
        关键说明：该函数支持在复杂架构中自定义训练的权重组和权重组所对应的学习率

            使用方法：
            class Net(NNFrameWork):
                def __init__(self):
                    super().__init__()
                    self.update_mark = True # or False
                    self.lr = 0.1
            # 此时，该模块内的权重将被设置为self.lr(包括该模块内部没有设置学习率的子模块), 是否更新则按照update_mark的设置进行

        :return: params(list)
        """
        params, _, _ = get_params(self)
        return params

    def set_weights_init(self, init_f):
        self.init_fs.append(init_f)
        self.apply(init_f)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_device(self, device=None):
        if device is None:
            # 检查 GPU 是否可用，如果可用则选择 cuda:0，否则选择 cpu
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device
        self.to(self.device)
