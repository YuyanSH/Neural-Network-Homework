import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 预处理
# 数据增强：对训练集使用随机裁剪 (RandomCrop) 和随机水平翻转 (RandomHorizontalFlip) 等方法进行扩充，可有效提高模型泛化能力
# 。例如，对 32×32 的 CIFAR-10 图像使用 RandomCrop(size=32, padding=4) 和 RandomHorizontalFlip()，并在最后转为张量 (ToTensor)。
# 归一化：对训练集和测试集图像均进行标准化（减均值除以标准差）。
# CIFAR-10 常用的标准化参数为均值 (0.4914,0.4822,0.4465)，标准差 (0.247,0.243,0.261)
#使像素值落在类似均匀分布区间，有助于模型稳定训练。

# 定义训练集的变换：随机裁剪、随机水平翻转，并转换为 Tensor、标准化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),                    # 随机水平翻转:
    transforms.RandomCrop(32, padding=4),                 # 随机裁剪到 32x32，并进行 4 像素填充
    transforms.ToTensor(),                                # 转为 [0,1] 之间的 Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465),        # 标准化：RGB均值 (0.4914,0.4822,0.4465)
                         (0.2470, 0.2435, 0.2616))        # RGB标准差 (0.2470,0.2435,0.2616)
])

# 定义验证/测试集的变换：只做中心裁剪 (可选) 和标准化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

# 下载并加载 CIFAR-10 训练数据集（用于训练和验证），无变换
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# 将训练集分为训练集和验证集（如 80% 训练，20% 验证）
num_train = int(0.8 * len(full_train_dataset))
num_val = len(full_train_dataset) - num_train
torch.manual_seed(42)
indices = torch.randperm(len(full_train_dataset)).tolist()
train_indices, val_indices = indices[:num_train], indices[num_train:]

# 创建训练/验证数据子集，分别附加对应的变换
train_dataset = Subset(torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform), train_indices)
val_dataset   = Subset(torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform),  val_indices)

# 加载数据
batch_size = 128  # 可根据显存调整，保证不超过16GB
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

# 测试集
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# 模型定义
# 自定义 CNN：实现一个简单的卷积神经网络，包括多个卷积层、ReLU 激活和最大池化层，最后使用全连接层分类。
# 预训练 ResNet18：作为替代方案，可使用 torchvision.models.resnet18(pretrained=True) 进行迁移学习

# 。由于 CIFAR-10 图像为 32×32，可直接使用 ResNet18（或根据需要修改 conv1 和 maxpool）。
# 将最后的全连接层 fc 改为输出 10 类。

import torchvision.models as models

# 自定义卷积神经网络
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输出 32@32x32
        self.pool  = nn.MaxPool2d(2, 2)                          # 池化后尺寸减半 (16x16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输出 64@16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# 输出 128@16x16

        # 全连接层：需要先将特征图展平为向量
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 假设经过两次池化后特征图为 8x8
        self.fc2 = nn.Linear(256, 10)           # 输出 10 类

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)                      # 第一轮池化，尺寸变为 16x16
        x = torch.relu(self.conv2(x))
        x = self.pool(x)                      # 第二轮池化，尺寸变为 8x8
        x = torch.relu(self.conv3(x))
        # 将多通道特征图展平为向量（保留 batch 维度）
        x = torch.flatten(x, 1)               # 等价于 x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x))                      
        return x

# 实例化模型：可以在此切换使用 CustomCNN 或 ResNet18
use_resnet = True  # 设置为 True 则使用预训练 ResNet18，否则使用自定义网络

if use_resnet:
    model = models.resnet18(pretrained=True)
    # 修改最后一层全连接层输出为 10 类
    model.fc = nn.Linear(model.fc.in_features, 10)
else:
    model = CustomCNN()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# 模型训练
# 损失函数与优化器：采用交叉熵损失 (nn.CrossEntropyLoss) 和 带动量的SGD 优化器
# 例如，学习率设为 0.001，动量设为 0.9
# 训练过程：对于每个训练轮次（epoch），在训练集上迭代进行前向传播、反向传播并更新参数。
# 同时计算并累加训练损失。每轮结束后，在验证集上评估模型：计算验证损失和验证准确率，并记录。
# 记录指标：使用列表 train_losses、val_losses 和 val_accuracies 来存储每个 epoch 的训练损失、验证损失和验证准确率，后续绘图。

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 使用 SGD 
num_epochs = 30  # 设定训练轮次，可根据需要调整
train_losses = []
val_losses   = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 训练阶段
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4%}")



# 模型评估
# 测试集评估：在完整的测试集上评估模型性能。记录所有测试样本的预测结果和真实标签，并计算整体准确率。
# 这可以帮助分析哪些类别表现较好或较差。
# 每类精度：可以单独计算并输出每个类别的精度，例如使用 precision_score（average=None）或从分类报告中提取。

# 分类报告：使用 scikit-learn 的 classification_report 输出每个类别的精确率 (Precision)、召回率 (Recall) 和 F1 分数


from sklearn.metrics import classification_report, precision_score

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        max_val, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # print(len(all_preds))
    # print(len(all_labels))

# 计算整体准确率
test_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"Overall Test Accuracy: {test_accuracy:.4%}")


class_names = test_dataset.classes  # CIFAR-10 类别名称
# 如需单独输出每类精度，可使用 precision_score
precisions = precision_score(all_labels, all_preds, average=None)
for cls, p in zip(class_names, precisions):
    print(f"Precision of {cls}: {p:.2%}")


# 输出分类报告（包含每类精度、召回率）
# print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=class_names))


import matplotlib.pyplot as plt

epochs = range(1, num_epochs+1)

plt.figure(figsize=(12,4))
# 子图1：训练损失和验证损失
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'r-', label='Training Loss')
plt.plot(epochs, val_losses,   'b-', label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 子图2：验证准确率
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig('/root/homework2/training_metrics.png')
