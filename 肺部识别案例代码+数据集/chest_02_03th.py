import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# 获取当前代码所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置 TORCH_HOME 环境变量为当前目录下的 'torch_cache' 文件夹
os.environ['TORCH_HOME'] = os.path.join(current_dir, 'torch_cache')

# 2 定义一个方法：显示图片
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# 8 更改池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)  # 池化层的卷积核大小，默认值为（1，1）
        self.pool_one = nn.AdaptiveAvgPool2d(size)  # 池化层1
        self.pool_two = nn.AdaptiveAvgPool2d(size)  # 池化层2

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)], 1)  # 连接两个池化层


# 7 迁移学习：拿到一个成熟的模型，进行模型微调
# 迁移学习：拿到一个成熟的模型，进行模型微调
def get_model():
    from torchvision.models import ResNet50_Weights  # 导入权重枚举
    # 获取预训练模型
    model_pre = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # 冻结预训练模型中所有的参数
    for param in model_pre.parameters():#model_pre.parameters() 返回一个迭代器，该迭代器给出了模型中所有的参数
        # param.requires_grad 是 PyTorch 中每个参数对象的一个属性，它决定了该参数是否需要计算梯度。
        # 如果设置为 False，那么即使在反向传播过程中，也不会计算这个参数的梯度，进而不会更新该参数的值。
        # 将它们的 requires_grad 属性设为 False，可以确保在后续的训练过程中，这些参数不会被优化算法更新
        param.requires_grad = False

    # 微调模型：替换 ResNet 最后的两层网络，返回一个新的模型
    model_pre.avgpool = AdaptiveConcatPool2d()  # 池化层替换
    model_pre.fc = nn.Sequential(
        nn.Flatten(),  # 所有维度拉平
        nn.BatchNorm1d(4096),  # 256 x 6 x 6 ——> 4096
        nn.Dropout(0.5),  # 丢掉一些神经元
        nn.Linear(4096, 512),  # 线性层的处理
        nn.ReLU(),  # 激活层
        nn.BatchNorm1d(512),  # 正则化处理
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1),  # 损失函数
    )

    return model_pre



def main():
    # 3 定义超参数
    BATCH_SIZE = 8  # 每批处理的数据数量

    # 检查是否有可用的 GPU，并选择相应设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 如果使用 GPU，显示更多 GPU 信息
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 4 图片转换
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),

        'val':
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    # 5 操作数据集
    # 5.1 数据集路径
    data_path = "H:\\pytorch_study\\轻松学PyTorch\\p4_肺部感染识别案例\\肺部识别案例代码+数据集\\chest_xray"
    # 5.2 加载数据集train 和 val
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    # 5.3 为数据集创建一个迭代器，读取数据
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True,
                                 batch_size=BATCH_SIZE) for x in ['train', 'val']}

    # 5.3 训练集和验证集的大小（图片的数量）
    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 5.4 获取标签的类别名称:  NORMAL 正常 --- PNEUMONIA 感染
    target_names = image_datasets['train'].classes

    # 6 显示一个batch_size的图片（8张图片）
    # 6.1 读取8张图片
    datas, targets = next(iter(dataloaders['train']))
    # 6.2 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 6.3 显示图片
    image_show(out, title=[target_names[x] for x in targets])

    # 7 加载模型并转移到 DEVICE
    model = get_model().to(DEVICE)
    print("Model moved to:", next(model.parameters()).device)

    # 8 示例：将数据转移到 DEVICE 并进行简单前向传播
    datas, targets = datas.to(DEVICE), targets.to(DEVICE)
    print("Data moved to:", datas.device)
    with torch.no_grad():
        outputs = model(datas)
    print(outputs)


if __name__ == '__main__':
    main()
