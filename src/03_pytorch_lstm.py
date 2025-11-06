# src/03_pytorch_lstm.py

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random

# 导入工具函数
from utils import evaluate_model_performance, reshape_for_lstm

# 定义文件路径和常量
PROCESSED_DATA_PATH = 'data/processed'
X_train_file = os.path.join(PROCESSED_DATA_PATH, 'X_train.csv')
X_test_file = os.path.join(PROCESSED_DATA_PATH, 'X_test.csv')
y_train_file = os.path.join(PROCESSED_DATA_PATH, 'y_train.csv')
y_test_file = os.path.join(PROCESSED_DATA_PATH, 'y_test.csv')

# --- 1. 配置 ---
# 确保实验可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# 模型超参数
NUM_ORIGINAL_FEATURES = 44 # 原始监测数据的特征数量 (F)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 100 # 早停容忍次数
DROPOUT_RATE = 0.2

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# --- 2. PyTorch 模型定义 ---

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(64, output_size)

    def forward(self, x):
        # x 形状: [batch_size, time_steps, input_size]
        out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一层 LSTM 的最终隐藏状态作为序列表示
        final_h = h_n[-1, :, :] 
        
        out = self.fc1(final_h)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out

# --- 3. 训练与评估函数 ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    """PyTorch 模型训练循环，包含早停机制。"""
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train() 
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证集评估
        val_loss = validate_model(model, val_loader, criterion)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # 最佳模型状态保存 (如果需要)
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.6f}, Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f"早停触发于 Epoch {epoch+1}")
            break

def validate_model(model, val_loader, criterion):
    """计算验证集损失。"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

def evaluate_and_predict(model, test_loader):
    """在测试集上进行预测和评估。"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            all_preds.append(outputs)
            all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    evaluate_model_performance(y_true, y_pred, "PyTorch LSTM Regressor")

# --- 4. 主流程 ---

if __name__ == '__main__':
    print("--- 1. 数据加载与预处理 ---")
    try:
        X_train_df = pd.read_csv(X_train_file)
        X_test_df = pd.read_csv(X_test_file)
        y_train_df = pd.read_csv(y_train_file)
        y_test_df = pd.read_csv(y_test_file)
    except FileNotFoundError:
        print("错误：找不到处理后的数据。请先运行 src/01_data_prep.py！")
        exit()

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    # 重塑为 3D 形状 [样本数, N_max, F]
    X_train_reshaped = reshape_for_lstm(X_train_scaled, NUM_ORIGINAL_FEATURES)
    X_test_reshaped = reshape_for_lstm(X_test_scaled, NUM_ORIGINAL_FEATURES)
    
    # 转换为 PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

    # 划分训练集和验证集 (使用 90% 训练，10% 验证)
    train_size = int(0.9 * len(X_train_tensor))
    val_size = len(X_train_tensor) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(X_train_tensor, y_train_tensor), [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    print(f"  -> 训练集/验证集/测试集大小: {train_size}/{val_size}/{len(X_test_df)}")

    # 模型实例化
    model = LSTMRegressor(
        input_size=NUM_ORIGINAL_FEATURES, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- 2. 开始模型训练 ---")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=PATIENCE)

    print("\n--- 3. 测试集评估 ---")
    # 由于测试集 DataLoader 只有一个 X_test_tensor，我们需要一个简单的测试集 DataLoader
    test_target_tensor = torch.tensor(y_test_df.values, dtype=torch.float32)
    test_loader_full = DataLoader(TensorDataset(X_test_tensor, test_target_tensor), batch_size=BATCH_SIZE, shuffle=False)
    
    evaluate_and_predict(model, test_loader_full)