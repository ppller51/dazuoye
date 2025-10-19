import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import glob
import numpy as np

# 工作空间路径设置
WORKSPACE = '/home/pl/5/shuzi_juanji'
DATASET_DIR = os.path.join(WORKSPACE, 'datasets')
MODEL_SAVE_DIR = WORKSPACE

# 支持的图像格式
SUPPORTED_FORMATS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')

# 类别文件夹名称
CLASS_FOLDERS = ['1', '2', '3', '4', '5', '6outpost', '7guard', '8base', '9neg']

# 自定义二值化转换 - 与C++中的OTSU算法保持一致
class BinaryTransform:
    def __init__(self, threshold=0):
        self.threshold = threshold
        
    def __call__(self, img):
        # 转换为灰度图（与C++一致）
        img_gray = transforms.functional.to_grayscale(img)
        
        # 转换为numpy数组进行处理
        img_np = np.array(img_gray, dtype=np.uint8)
        
        # 使用大律法(OTSU)自动计算阈值
        if self.threshold == 0:
            # 实现OTSU算法
            hist = np.histogram(img_np, bins=256, range=(0, 256))[0]
            total = img_np.size
            sum_total = np.sum(np.arange(256) * hist)
            sum_bg = 0
            w_bg = 0
            w_fg = 0
            max_var = 0
            threshold = 0
            
            for t in range(256):
                w_bg += hist[t]
                if w_bg == 0:
                    continue
                w_fg = total - w_bg
                if w_fg == 0:
                    break
                    
                sum_bg += t * hist[t]
                mean_bg = sum_bg / w_bg
                mean_fg = (sum_total - sum_bg) / w_fg
                
                var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
                
                if var_between > max_var:
                    max_var = var_between
                    threshold = t
        else:
            threshold = self.threshold
            
        # 应用阈值进行二值化
        img_binary = (img_np >= threshold) * 255
        img_binary = img_binary.astype(np.uint8)
        
        # 转换回PIL图像
        img_pil = transforms.functional.to_pil_image(img_binary)
        
        return img_pil

# 数据预处理 - 与C++保持一致
transform = transforms.Compose([
    transforms.Resize((240, 100)),  # 调整为100x240 (宽x高)
    transforms.GaussianBlur(kernel_size=5),  # 自动计算sigma，与OpenCV一致
    BinaryTransform(),  # 二值化处理（OTSU算法）
    transforms.ToTensor(),
    # 转为3通道（复制单通道三次），与C++中处理一致
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 自定义数据集加载器
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        
    def is_valid_file(self, filename):
        return filename.lower().endswith(SUPPORTED_FORMATS)

# 检查工作空间和数据集
def check_workspace():
    if not os.path.exists(WORKSPACE):
        raise FileNotFoundError(f"工作空间不存在: {WORKSPACE}")
    
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"数据集目录不存在: {DATASET_DIR}")
    
    for folder in CLASS_FOLDERS:
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"警告: 类别文件夹不存在 - {folder_path}")
            continue
            
        png_files = glob.glob(os.path.join(folder_path, '*.png')) + \
                   glob.glob(os.path.join(folder_path, '*.PNG'))
                   
        if len(png_files) == 0:
            print(f"警告: 类别 {folder} 中未找到PNG图像")
        else:
            print(f"找到 {folder} 类别PNG图像 {len(png_files)} 张")

# 执行工作空间检查
try:
    check_workspace()
except FileNotFoundError as e:
    print(f"错误: {e}")
    exit(1)

# 加载数据集
full_dataset = CustomImageFolder(root=DATASET_DIR, transform=transform)

# 显示类别映射
print("\n类别映射关系:")
for class_name, class_idx in full_dataset.class_to_idx.items():
    print(f"文件夹 '{class_name}' -> 标签 {class_idx}")

# 数据集划分
total_samples = len(full_dataset)
if total_samples == 0:
    print("错误: 未找到任何有效图像文件")
    exit(1)

train_size = max(int(0.8 * total_samples), 1)
val_size = total_samples - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 数据加载器
batch_size = min(32, train_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据集统计: 总样本={total_samples}, 训练集={train_size}, 验证集={val_size}")

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32x120x50
        x = self.pool(torch.relu(self.conv2(x)))  # 64x60x25
        x = self.pool(torch.relu(self.conv3(x)))  # 128x30x12
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
print("\n开始训练...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total if val_total > 0 else 0
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'训练: 损失={epoch_loss:.4f}, 准确率={epoch_acc:.2f}%')
    print(f'验证: 准确率={val_acc:.2f}%\n')

# 保存模型
pth_path = os.path.join(MODEL_SAVE_DIR, 'cnn_model_100x240_binary.pth')
torch.save(model.state_dict(), pth_path)
print(f"PyTorch二值化模型已保存至: {pth_path}")

# 导出ONNX - 使用传统导出方式（移除dynamo参数）
onnx_path = os.path.join(MODEL_SAVE_DIR, 'cnn_model_100x240_binary.onnx')
model.eval()
dummy_input = torch.randn(1, 3, 240, 100)  # 匹配100x240输入

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11  # 传统导出方式，不依赖onnxscript
)

print(f"ONNX二值化模型已保存至: {onnx_path}")
print("完成! 模型与C++二值化处理完全对应，可直接用于OpenCV DNN推理")
