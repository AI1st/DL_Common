import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from IPython.display import display, clear_output
from ipywidgets import widgets
import matplotlib.pyplot as plt
import numpy as np


class ErrorVisualizer:
    """
    examples with cifar10:
    # 假设你已经有一个训练好的模型和测试数据加载器
    model = model  # 你的模型
    test_loader = test_iter  # 你的测试数据加载器
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 数据集的类别名称

    # 预处理列表
    trans_test = [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2020))
    ]

    # 创建错误可视化器
    visualizer = ErrorVisualizer(model, test_loader, class_names, device='cuda', image_size=(2, 2), transform_list=trans_test)

    # 运行错误可视化器
    visualizer.run()
    """
    def __init__(self, model, test_loader, class_names, device='cpu', image_size=(8, 8), transform_list=None):
        """
        初始化错误可视化器
        :param model: 已训练的模型
        :param test_loader: 测试数据加载器
        :param class_names: 类别名称列表
        :param device: 设备（'cpu' 或 'cuda'）
        :param image_size: 显示图像的大小，默认为 (8, 8)
        :param transform_list: 预处理列表
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.errors = []
        self.current_index = 0
        self.image_size = image_size
        self.transform_list = transform_list

    def find_errors(self):
        """找到测试集中的错误样本"""
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
                for idx in incorrect_indices:
                    self.errors.append({
                        'image': images[idx].cpu(),
                        'label': labels[idx].item(),
                        'predicted': predicted[idx].item(),
                        'probabilities': probabilities[idx].cpu().numpy()
                    })

    def reverse_transform(self, image):
        """根据预处理列表进行逆变换"""
        if self.transform_list is not None:
            for t in reversed(self.transform_list):
                if isinstance(t, transforms.Normalize):
                    mean = torch.tensor(t.mean).view(-1, 1, 1)
                    std = torch.tensor(t.std).view(-1, 1, 1)
                    image = image * std + mean
                elif isinstance(t, transforms.ToTensor):
                    image = image.numpy().transpose(1, 2, 0)  # CHW to HWC
                elif isinstance(t, transforms.Resize):
                    image = transforms.ToPILImage()(image)
                    image = t(image)
                    image = np.array(image)
        return image

    def display_error(self, index):
        """显示特定索引的错误样本及其前五个预测概率和正确类别的概率"""
        error = self.errors[index]
        image = error['image']
        label = error['label']
        predicted = error['predicted']
        probabilities = error['probabilities']

        # 获取前五个预测概率及其类别名称
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_probabilities = probabilities[top5_indices]
        top5_classes = [self.class_names[i] for i in top5_indices]

        # 获取正确类别的概率
        true_class_probability = probabilities[label]

        # 转换图像形状并进行逆变换
        image = self.reverse_transform(image)

        # 显示图像
        plt.figure(figsize=self.image_size)
        plt.imshow(image)
        plt.title(f"True Label: {self.class_names[label]}, Predicted: {self.class_names[predicted]}")
        plt.axis('off')
        plt.show()

        # 显示前五个预测概率及其类别名称
        print("Top 5 Predictions:")
        for prob, cls in zip(top5_probabilities, top5_classes):
            print(f"{cls}: {prob:.4f}")

        # 显示正确类别的概率
        print(f"\nTrue Class Probability: {self.class_names[label]}: {true_class_probability:.4f}")

    def update_display(self):
        """更新显示"""
        clear_output(wait=True)  # 清除之前的输出
        if len(self.errors) > 0:
            self.current_index = self.current_index % len(self.errors)
            self.display_error(self.current_index)
            print(f"\nCurrent Index: {self.current_index + 1} of {len(self.errors)}")
        else:
            print("No errors found.")

    def run(self):
        """运行错误可视化器"""
        self.find_errors()
        while True:
            self.update_display()
            key = input("Press 'p' for previous, 'n' for next, 's' to set image size, or 'q' to quit: ")
            if key == 'p':
                self.current_index -= 1
            elif key == 'n':
                self.current_index += 1
            elif key == 's':
                try:
                    width = float(input("Enter new width (in inches): "))
                    height = float(input("Enter new height (in inches): "))
                    self.image_size = (width, height)
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            elif key == 'q':
                break
            else:
                print("Invalid input. Please press 'p' for previous, 'n' for next, 's' to set image size, or 'q' to quit.")