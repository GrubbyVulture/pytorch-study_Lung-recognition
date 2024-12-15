import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# 设置 TORCH_HOME 环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = os.path.join(current_dir, 'torch_cache')

# 定义一个方法：显示图片
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

# 自定义池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)
        self.pool_one = nn.AdaptiveAvgPool2d(size)
        self.pool_two = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)], 1)

# 获取微调的 ResNet50 模型
def get_model():
    from torchvision.models import ResNet50_Weights
    model_pre = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model_pre.parameters():
        param.requires_grad = False

    model_pre.avgpool = AdaptiveConcatPool2d()
    model_pre.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1),
    )

    return model_pre

# 主函数
def main():
    # 超参数
    BATCH_SIZE = 1024

    # 使用 CPU
    DEVICE = torch.device('cpu')
    print(f"Using device: {DEVICE}")

    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(300),
            transforms.RandomResizedCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集加载
    data_path = "H:\\pytorch_study\\轻松学PyTorch\\p4_肺部感染识别案例\\肺部识别案例代码+数据集\\chest_xray"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=BATCH_SIZE, num_workers=4) for x in ['train', 'val']}
    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    target_names = image_datasets['train'].classes

    # 显示一个 batch 的图片
    datas, targets = next(iter(dataloaders['train']))
    out = make_grid(datas, nrow=4, padding=10)
    image_show(out, title=[target_names[x] for x in targets])

    # 记录开始时间
    start_time = time.time()

    # 模型加载和前向传播示例
    model = get_model().to(DEVICE)
    print("Model moved to:", next(model.parameters()).device)

    datas, targets = datas.to(DEVICE), targets.to(DEVICE)
    print("Data moved to:", datas.device)

    with torch.no_grad():
        outputs = model(datas)
    print(outputs)

    # 记录结束时间
    end_time = time.time()
    print(f"Total execution time on CPU: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
