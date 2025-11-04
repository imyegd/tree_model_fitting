import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"识别到的 GPU 设备名: {torch.cuda.get_device_name(0)}")