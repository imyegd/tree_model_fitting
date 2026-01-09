import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 设备配置 (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 数据加载与预处理
df = pd.read_csv('data/raw/束流.csv', parse_dates=['时间'])
feature_cols = [f'feature{i}' for i in range(1, 36)]
target_col = 'target'

# 归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# 3. 构造滑动窗口数据
def create_sequences(X, y, seq_length=50):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

SEQ_LENGTH = 50 
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# 转换为 PyTorch 张量
X_tensor = torch.from_numpy(X_seq).float()
y_tensor = torch.from_numpy(y_seq).float()

# 划分训练集 (前15000条作为基准)
train_size = 15000
train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4. 定义 LSTM 模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

model = LSTMRegressor(input_dim=35, hidden_dim=64, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 模型训练
print("开始训练 PyTorch LSTM 基准模型...")
model.train()
for epoch in range(30): # 根据损失情况调整 epoch
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/30], Loss: {loss.item():.6f}')

# 6. 全量预测与残差计算
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_tensor.to(device)).cpu().numpy()

# 反向归一化还原真实物理量
y_actual = scaler_y.inverse_transform(y_seq)
y_pred = scaler_y.inverse_transform(predictions_scaled)

# 计算残差 (Residual)
residuals = np.abs(y_actual - y_pred).flatten()

# 7. 基于统计学打标 (3-Sigma)
# 只用训练段的残差来计算基准阈值
train_res = residuals[:train_size]
threshold = np.mean(train_res) + 3 * np.std(train_res)
is_anomaly = (residuals > threshold).astype(int)

# 8. 结果可视化
plt.figure(figsize=(15, 6))
time_axis = df['时间'].iloc[SEQ_LENGTH:]
plt.plot(time_axis, y_actual, label='Actual Target', alpha=0.7)
plt.plot(time_axis, y_pred, label='LSTM Predicted', linestyle='--', alpha=0.8)

# 标出异常点
plt.scatter(time_axis[is_anomaly == 1], y_actual[is_anomaly == 1], 
            color='red', s=10, label='Anomaly Detected')

plt.axhline(y=threshold, color='green', linestyle=':', label='Statistical Threshold')
plt.title('PyTorch LSTM Anomaly Detection (Residual Method)')
plt.legend()
plt.show()

print(f"检测完成，阈值设定为: {threshold:.4f}")