import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. 数据加载 (假设文件已保存) ---
try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
except FileNotFoundError:
    print("错误：请确保 X_train.csv, X_test.csv, y_train.csv, y_test.csv 已经生成并位于当前目录下。")
    # 如果文件不存在，这里应该重新执行数据准备逻辑，但为了简洁，我们假设用户已经执行了上一步的保存操作。
    # 实际项目中，如果文件不存在，应该返回错误或重新生成。
    exit()

# 转换为 NumPy 数组
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# 2. 标准化 (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# 3. 数据重塑 (Reshaping)
NUM_TOTAL_FEATURES = X_train_scaled.shape[1]
NUM_ORIGINAL_FEATURES = 44  # 原始监测数据的特征数量
TIME_STEPS = int(NUM_TOTAL_FEATURES / NUM_ORIGINAL_FEATURES)

print(f"数据重塑参数: 时间步长 (N_max) = {TIME_STEPS}, 单记录特征数 = {NUM_ORIGINAL_FEATURES}")

# [样本数, 总特征数] -> [样本数, 时间步长, 特征数]
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], TIME_STEPS, NUM_ORIGINAL_FEATURES)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], TIME_STEPS, NUM_ORIGINAL_FEATURES)

# 4. 转换为 PyTorch Tensors
# 确保数据类型为 float32
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

# 5. 创建 DataLoader
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 确定运行设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")