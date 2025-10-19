import torch
# 替换your_model_file为实际的模型定义文件名（不含.py扩展名）
from model import CNN  # 假设模型定义在model.py中

# 加载已训练好的模型（使用绝对路径更可靠）
model = CNN()
model_path = "/home/pl/5/shuzi_juanji/cnn_model_100x240.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# 导出ONNX
dummy_input = torch.randn(1, 3, 240, 100)
torch.onnx.export(
    model,
    dummy_input,
    'cnn_model_100x240.onnx',
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
