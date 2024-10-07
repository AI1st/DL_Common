import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML


class WeightMatrixVisualizer:
    """
    用于可视化神经网络权重矩阵随时间变化的3D散点图。

    参数:
    - input_size: 输入神经元的数量
    - output_size: 输出神经元的数量
    - weight_history: 包含权重历史的列表，每个元素是一个形状为 (output_size, input_size) 的二维数组
    - figsize: 图像的大小，默认为 (5, 5)
    - interval: 动画帧之间的间隔时间（毫秒），默认为 200
    """

    def __init__(self, input_size, output_size, weight_history, figsize=(5, 5), interval=200):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_history = np.array(weight_history)
        self.figsize = figsize
        self.interval = interval
        self.xlabel = 'Input Neuron Index'
        self.ylabel = 'Output Neuron Index'
        self.zlabel = 'Weight Value'
        self.title = 'Weight Matrix Evolution'

    def set_graph_features(self, xlabel, ylabel, zlabel, title):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.title = title

    def _create_3d_animation(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 创建网格
        x_grid, y_grid = np.meshgrid(np.arange(self.input_size), np.arange(self.output_size))
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        # 初始化3D散点
        scat = ax.scatter(x_flat, y_flat, self.weight_history[0].flatten(), c=self.weight_history[0].flatten(),
                          cmap='coolwarm')

        # 设置坐标轴标签
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)
        ax.set_title(self.title)

        def animate_diff(i, store):
            print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
            z_values = store[i].flatten()
            scat._offsets3d = (x_flat, y_flat, z_values)
            scat.set_array(z_values)

            # 动态调整z轴范围
            ax.set_zlim(np.min(z_values), np.max(z_values))

            return [scat]

        ani = animation.FuncAnimation(fig, animate_diff, fargs=[self.weight_history], interval=self.interval,
                                      blit=False, repeat=True, frames=self.weight_history.shape[0])

        return ani, fig

    def show_animation(self, save_path=None):
        """
        在Jupyter Notebook中显示动画，并可选择保存为GIF文件。

        参数:
        - save_path: 保存GIF文件的路径，如果为None则不保存
        """
        ani, fig = self._create_3d_animation()

        if save_path is not None:
            # 确保保存路径的目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            ani.save(save_path, writer='pillow', fps=10)
            print(f'Saved animation to {save_path}')

        html_ani = HTML(ani.to_jshtml())
        plt.close(fig)
        plt.ion()

        return html_ani

    def save_animation(self, save_path):
        """
        仅保存动画为GIF文件，不在Jupyter Notebook中显示。

        参数:
        - save_path: 保存GIF文件的路径
        """
        ani, fig = self._create_3d_animation()
        ani.save(save_path, writer='pillow', fps=10)
        plt.close(fig)
        print(f'Saved animation to {save_path}')


# 调用实例
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 创建一个简单的线性模型
    input_size = 10
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.Linear(output_size, output_size))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 生成一些示例数据
    x = torch.randn(100, input_size)
    y = torch.randn(100, output_size) + (x + 5) ** 2

    # 记录权重变化
    weight_history = []

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # 记录权重
        weight_history.append(model[0].weight.data.clone().detach().numpy())

    # 将权重历史转换为NumPy数组
    weight_history = np.array(weight_history)

    # 创建并显示动画
    visualizer = WeightMatrixVisualizer(input_size, output_size, weight_history)
    ani = visualizer.show_animation(save_path='examples/weight_matrix_evolution.gif')