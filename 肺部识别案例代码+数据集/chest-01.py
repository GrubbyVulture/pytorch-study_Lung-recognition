"""
这份代码是在cpu上计算的，因为计算的数据并没转移到DEVICE上
只是显示图片，没有调用模型进行计算
"""

# 1 加载库
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 2 定义一个方法：显示图片（官网给的方法）
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def main():
    # 3 定义超参数
    BATCH_SIZE = 8 # 每批处理的数据数量
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4 图片转换
    # 定义图像预处理转换的字典，分别针对训练集和验证集
    data_transforms = {
        'train': transforms.Compose([
            # 将输入图像的最小边调整为 300 像素，同时保持原始宽高比。
            # 这一步骤确保所有输入图像至少有一边是 300 像素大小。
            transforms.Resize(300),

            # 随机裁剪图像并调整其大小为 300x300 像素。
            # 这个操作会从原图中随机选取一个区域进行裁剪，并将该区域缩放至指定大小。
            # 这有助于增加数据多样性，提高模型的泛化能力。
            transforms.RandomResizedCrop(300),

            # 以 50% 的概率随机水平翻转图像。
            # 这也是数据增强的一种方式，可以帮助模型学习到对称性特征，从而更好地泛化到新数据上。
            transforms.RandomHorizontalFlip(),

            # 对图像进行中心裁剪，使其最终尺寸为 256x256 像素。
            # 这一步确保了所有输入到模型中的图像都具有相同的尺寸，同时也移除了边缘部分，可能减少了背景信息的影响。
            transforms.CenterCrop(256),

            # 将 PIL 图像或 NumPy 数组转换为 PyTorch 的张量（Tensor）。
            # 转换后，图像的像素值会被归一化到 [0, 1] 范围内，并且通道顺序会从 HWC (Height, Width, Channel) 变为 CHW (Channel, Height, Width)，这是 PyTorch 所需的格式。
            transforms.ToTensor(),

            # 使用给定的均值 [0.485, 0.456, 0.406] 和标准差 [0.229, 0.224, 0.225] 来标准化图像张量。
            # 这里的数值通常是基于 ImageNet 数据集计算得出的，它们分别对应于 RGB 三个通道的统计值。
            # 标准化可以加速模型收敛，因为它使得不同特征具有相似的数据分布。
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            # 将输入图像的最小边调整为 300 像素，同时保持原始宽高比。
            # 这一步骤确保所有输入图像至少有一边是 300 像素大小。
            transforms.Resize(300),

            # 对图像进行中心裁剪，使其最终尺寸为 256x256 像素。
            # 这一步确保了所有输入到模型中的图像都具有相同的尺寸，同时也移除了边缘部分，可能减少了背景信息的影响。
            transforms.CenterCrop(256),

            # 将 PIL 图像或 NumPy 数组转换为 PyTorch 的张量（Tensor）。
            # 转换后，图像的像素值会被归一化到 [0, 1] 范围内，并且通道顺序会从 HWC (Height, Width, Channel) 变为 CHW (Channel, Height, Width)，这是 PyTorch 所需的格式。
            transforms.ToTensor(),

            # 使用给定的均值 [0.485, 0.456, 0.406] 和标准差 [0.229, 0.224, 0.225] 来标准化图像张量。
            # 这里的数值通常是基于 ImageNet 数据集计算得出的，它们分别对应于 RGB 三个通道的统计值。
            # 标准化可以加速模型收敛，因为它使得不同特征具有相似的数据分布。
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 5 操作数据集
    # 5.1 数据集路径
    data_path = "H:\\pytorch_study\\轻松学PyTorch\\p4_肺部感染识别案例\\肺部识别案例代码+数据集\\chest_xray"
    # 5.2 加载数据集train 和 val
    image_datasets = { x : datasets.ImageFolder(os.path.join(data_path, x),
                                                data_transforms[x]) for x in ['train', 'val']}
    # 5.3 为数据集创建一个迭代器，读取数据
    datalaoders = {x : DataLoader(image_datasets[x], shuffle=True,
                                  batch_size=BATCH_SIZE) for x in ['train', 'val']}

    # 5.3 训练集和验证集的大小（图片的数量）
    data_sizes = {x : len(image_datasets[x]) for x in ['train', 'val']}

    # 5.4 获取标签的类别名称:  NORMAL 正常 --- PNEUMONIA 感染
    target_names = image_datasets['train'].classes

    # 6 显示一个batch_size的图片（8张图片）
    # 6.1 读取8张图片
    datas, targets = next(iter(datalaoders['train']))
    # 6.2 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 6.3 显示图片
    image_show(out, title=[target_names[x] for x in targets])


if __name__ == '__main__':
    main()





