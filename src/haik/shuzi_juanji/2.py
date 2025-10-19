import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# 定义与训练时相同的模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 使用全局平均池化（GAP）来应对任意尺寸的输入图像
        self.gap = nn.AdaptiveAvgPool2d(1)  # 输出尺寸将是 (batch_size, 128, 1, 1)
        self.fc1 = nn.Linear(128, 128)  # fc1 的输入维度是 128，因为输出通道数是 128
        self.fc2 = nn.Linear(128, 9)  # 输出9个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.gap(x)  # 全局平均池化，将特征图压缩为 (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)  # 展平 (batch_size, 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化模型
model = CNN()

# 加载保存的模型权重
model.load_state_dict(torch.load('/home/pl/5/shuzi_juanji/cnn_model.pth'))
model.eval()  # 设置为评估模式

# 图片预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载一张图片并预处理
image_path = r'/home/pl/5/shuzi_juanji/datasets/3/1275493643.png'  # 这里使用你自己的图片路径
image = Image.open(image_path)

# 如果是灰度图像，转换为RGB
if image.mode != 'RGB':
    image = image.convert('RGB')

# 对图像进行处理，使其适配模型的输入要求
image = transform(image)
image = image.unsqueeze(0)  # 扩展维度，batch_size=1

# 预测
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)

# 类别从1开始，因此加1
predicted_class += 1

print(f'预测的类别是：{predicted_class.item()}')
