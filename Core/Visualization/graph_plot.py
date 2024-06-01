import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline


def plot(x, ys, labels=None, xlabel="", ylabel="", title="", save=False, filepath='plot.png'):
    # 判断ys是否为容器类型变量(判断ys是否传入了多变量)
    backend_inline.set_matplotlib_formats('svg')
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
    if save:  # If save flag is True, save the figure to the provided filepath
        plt.savefig(filepath, format='png', dpi=300)
    plt.show()


def plot2(xs, ys, labels=None, xlabel="", ylabel="", title="", using_log=False,
          save=False, filepath='plot.png'):
    # 判断ys是否为容器类型变量(判断ys是否传入了多变量)
    backend_inline.set_matplotlib_formats('svg')
    if not isinstance(xs, (list, tuple, set, dict)):
        if not isinstance(ys, (list, tuple, set, dict)):
            xs = [xs]
            ys = [ys]
            labels = [labels]
        else:
            xs = [xs] * len(ys)
    else:
        if not isinstance(ys, (list, tuple, set, dict)):
            ys = [ys]
        assert len(xs) == len(ys), "xs do not match with ys!"

    # 判断标签是否为空
    if labels is None:
        labels = [""] * len(ys)
    fig, ax = plt.subplots()
    for x, y, label in zip(xs, ys, labels):
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)  # 设置x轴名称 x label
    ax.set_ylabel(ylabel)  # 设置y轴名称 y label
    ax.set_title(title)  # 设置图名为Simple Plot
    if using_log is True:
        ax.set_yscale('log')
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    if save:  # If save flag is True, save the figure to the provided filepath
        plt.savefig(filepath, format='png', dpi=300)
    plt.show()


def scatter(x, ys, labels=None, xlabel="", ylabel="", title=""):
    # 判断ys是否为容器类型变量(判断ys是否传入了多变量)
    backend_inline.set_matplotlib_formats('svg')
    if not isinstance(ys, (list, tuple, set, dict)):
        ys = [ys]
        labels = [labels]
    # 判断标签是否为空
    if labels is None:
        labels = [""] * len(ys)
    fig, ax = plt.subplots()
    for y, label in zip(ys, labels):
        ax.scatter(x, y, label=label)
    ax.set_xlabel(xlabel)  # 设置x轴名称 x label
    ax.set_ylabel(ylabel)  # 设置y轴名称 y label
    ax.set_title(title)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
