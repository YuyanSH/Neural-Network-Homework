from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

# net = ptcv_get_model("resnet18", pretrained=True)
# x = Variable(torch.randn(1, 3, 224, 224))
# y = net(x)

net = ptcv_get_model("resnet20_cifar10", pretrained=True)  # ✅ 输出为10类，输入为32x32

x = Variable(torch.randn(1, 3, 32, 32))  # ✅ 适配 CIFAR-10 图像大小
y = net(x)

print(y.shape)  # 应输出 torch.Size([1, 10])