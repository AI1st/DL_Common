import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data


def plot(x, ys, labels=None, xlabel="", ylabel="", title=""):
    # 判断ys是否为容器类型变量(判断ys是否传入了多变量)
    if not isinstance(ys, (list, tuple, set, dict)):
        ys = [ys]
        labels = [labels]
    # 判断标签是否为空
    if labels is None:
        labels = [""] * len(ys)
    fig, ax = plt.subplots()
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)  # 设置x轴名称 x label
    ax.set_ylabel(ylabel)  # 设置y轴名称 y label
    ax.set_title(title)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示


def train(data_set, model, loss, optimizer, epoch_num, batch_range=(50, 50), rand=False):
    """
    函数API按照数据、模型、损失、优化(数据流动的方式)、训练次数、训练batch范围、是否在迭代过程中随机生成迭代器的方式进行
    :param data_set: 由数据特征和数据标签构成
    :param model: 输入模型
    :param loss: 损失函数
    :param optimizer: 优化器
    :param epoch_num: 迭代次数
    :param batch_range: 单batch范围
    :param rand: 是否在迭代过程中随机生成迭代器
    :return: 返回训练的历史数据
    """
    l = None
    train_ls = []
    data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
    for epoch in range(epoch_num):
        if rand:
            data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
        for x, y in data_iter:
            optimizer.zero_grad()
            y_hat = model(x)
            l = loss(y, y_hat)
            l.mean().backward()
            optimizer.step()
        train_ls.append(l.detach().mean())
    return train_ls


class Accumulator:
    # 此类来自于d2l包，为累加器类
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    # 此类来自于d2l包，为计时器
    def __init__(self):
        self.times = []
        self.tik = None

        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时，并将时间加至列表"""
        tok = time.time()
        self.times.append(tok - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回总时间"""
        return sum(self.times)

    def cumsum(self):
        """返回累加时间列表"""
        return np.array(self.times).cumsum().tolist()


class Train:
    def __init__(self, model, optimizer, loss=None):
        """
        函数API按照模型、损失、优化(数据流动的方式)
        :param model: 输入模型
        :param loss: 损失函数
        :param optimizer: 优化器
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        # 参数初始化
        self.train_ls = []
        self.train_ls_temp = []
        self.total_number_of_samples = None
        self.times = []

    def __call__(self, data_set, epoch_num, batch_range=(50, 50), rand=False, test_set=None):  # train
        """
        :param data_set:由数据特征和数据标签构成
        :param epoch_num:迭代次数
        :param batch_range:单batch范围
        :param rand:是否在迭代过程中随机生成迭代器
        :param test_set:传入测试集
        :return: 训练历史记录
        """
        print("testing 03")  # 测试标记
        return self.train(data_set, epoch_num, batch_range, rand, test_set)

    def train(self, data_set, epoch_num, batch_range, rand=False, test_set=None):
        test_loss = ["undefined"]
        self.total_number_of_samples = len(data_set)
        # 训练
        timer = Timer()
        train_ls = self.multi_epoch(data_set, epoch_num, batch_range, rand, True)
        train_time = timer.stop()
        self.times = timer.times
        # 计算全局损失
        total_loss = self.multi_epoch(data_set, 1, len(data_set), False, False)
        # 计算测试集损失
        if test_set:
            test_loss = self.multi_epoch(test_set, 1, len(data_set), False, False)
        # 训练记录输出
        print(f"training loss: {total_loss[0]}, test loss: {test_loss[0]} training time: {train_time}")
        self.train_ls = [*self.train_ls, *train_ls]  # 合并训练记录
        self.train_ls_temp = train_ls
        self.plot_history()
        return self.train_ls

    def multi_epoch(self, data_set, epoch_num, batch_range, rand, train_flag):
        # 初始化及用户及初始判定
        train_ls = []
        if isinstance(batch_range, int):  # 若传入一个整数，则将其打包为元组
            batch_range = (batch_range, batch_range)
        # 获得数据集
        data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
        for epoch in range(epoch_num):
            if rand:  # 判断是否要重新生成数据集
                data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
            loss_in_one_epoch = self.single_epoch(data_iter, train_flag)
            train_ls.append(loss_in_one_epoch)
        return train_ls

    def single_epoch(self, data_iter, train_flag):
        metric = Accumulator(2)  # 第一项为累加次数，第二项为总误差
        for k in data_iter:
            batch_size = k[0].shape[0]  # 取样本的第一个维度
            l = self.single_batch(train_flag, *k) * batch_size  # 此时k为列表，通过*拆包
            metric.add(batch_size, l.detach().sum())
        loss_in_one_epoch = metric[1] / metric[0]
        return loss_in_one_epoch

    def single_batch(self, train_flag, *k):  # 不限输入的参数
        self.optimizer.zero_grad()
        l = self.loss_in_single_batch(*k)  # 此时k为元组，通过*k拆包
        if train_flag:
            l.mean().backward()
            self.optimizer.step()
        return l

    def loss_in_single_batch(self, x, y):  # 默认的计算损失方式
        """
        当需要修改最终的代价函数(例如增加正则项时可以重写此方法)
        :param x: 从迭代器中拿出来数据的第一维
        :param y: 从迭代器中拿出来数据的第二维
        :return: 损失
        """
        y_hat = self.model(x)
        l = self.loss(y, y_hat)
        return l

    def plot_history(self):
        plot(np.array(range(len(self.train_ls_temp))) + len(self.train_ls) - len(self.train_ls_temp) + 1,
             [self.train_ls_temp], xlabel="epoch", ylabel="train_loss",
             title=f"training history")
        plot(np.array(range(len(self.train_ls))) + 1, [self.train_ls], xlabel="epoch", ylabel="train_loss",
             title=f"training history")

    def reset_optimizer(self, optimizer):
        self.optimizer = optimizer

    def reset_loss(self, loss):
        self.loss = loss


if __name__ == "__main__":
    # 子类集成父类后方法改写测试
    class Parent:
        def __init__(self):
            pass

        def say(self, a):
            print(a)


    class Child():
        def __init__(self):
            pass

        def say(self, a, b):
            print(a, b)


    child = Child()
    child.say(1, 2)
