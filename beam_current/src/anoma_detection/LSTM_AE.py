import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 数据准备
df = pd.read_csv('data/raw/束流.csv')
# 将 target 和 features 放在一起作为一个整体向量
all_cols = ['target'] + [f'feature{i}' for i in range(1, 36)]
data = df[all_cols].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 构造序列 (Batch, Seq_Len, Feature_Dim)
def create_sequences(data, seq_len=50):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

SEQ_LEN = 50
X_seq = create_sequences(data_scaled, SEQ_LEN)
X_tensor = torch.from_numpy(X_seq).float()

# 训练集：取前15000个序列作为“正常基准”
train_tensor = X_tensor[:15000]
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)

# 2. 定义 LSTM-AutoEncoder 模型
class LSTM_AE(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(LSTM_AE, self).__init__()
        # Encoder: 压缩序列信息
        self.encoder = nn.LSTM(n_features, embedding_dim, num_layers=1, batch_first=True)
        # Decoder: 从压缩向量还原序列
        self.decoder = nn.LSTM(embedding_dim, n_features, num_layers=1, batch_first=True)
        
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.encoder(x) # 取最后的隐藏状态作为压缩表示
        
        # 准备 Decoder 的输入：将隐藏状态重复 SEQ_LEN 次
        # hidden shape: (1, batch, embedding_dim)
        repeat_hidden = hidden.repeat(SEQ_LEN, 1, 1).transpose(0, 1) 
        
        out, _ = self.decoder(repeat_hidden)
        return out # 返回重建后的序列

# 初始化模型 (输入36维，压缩到16维)
model = LSTM_AE(n_features=36, embedding_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. 训练过程
model.train()
for epoch in range(20):
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x) # 目标是让输出和输入一模一样
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 4. 异常检测 (计算重建误差)
# model.eval()
# with torch.no_grad():
#     X_reconstructed = model(X_tensor)
#     # 计算每个时间步的平均重建误差 (MSE)
#     # 我们只关注序列最后一个点的重建误差
#     diff = (X_reconstructed[:, -1, :] - X_tensor[:, -1, :]) ** 2
#     reconstruction_errors = torch.mean(diff, dim=1).numpy()

# 在你的代码第4步之后添加
model.eval()
with torch.no_grad():
    X_reconstructed = model(X_tensor)
    # 计算所有特征的综合重建误差 (MSE)
    mse_errors = torch.mean((X_reconstructed[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()

# 绘制误差曲线
plt.figure(figsize=(15, 5))
plt.plot(mse_errors, label='Reconstruction Error (Anomaly Score)', color='orange')
# 自动计算一个统计阈值（均值 + 3倍标准差）
threshold = np.mean(mse_errors[:15000]) + 3 * np.std(mse_errors[:15000])
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomaly Score (LSTM-AE Reconstruction Error)')
plt.xlabel('Time Step')
plt.ylabel('MSE Error')
plt.legend()
plt.show()

# 选择你感兴趣的异常时间段
start_idx, end_idx = 24000, 24150
# 计算每个特征的平方误差
detailed_errors = (X_reconstructed[start_idx:end_idx, -1, :] - X_tensor[start_idx:end_idx, -1, :])**2
detailed_errors_np = detailed_errors.numpy()

# 绘图
import seaborn as sns
plt.figure(figsize=(15, 8))
sns.heatmap(detailed_errors_np.T, yticklabels=all_cols, cmap='YlOrRd')
plt.title(f'Feature Contribution Heatmap (From {start_idx} to {end_idx})')
plt.xlabel('Relative Time Step')
plt.ylabel('Features')
plt.show()
# 5. 诊断：计算特征贡献度
# 误差最大的那个特征就是嫌疑人
# feature_errors = (X_reconstructed[:, -1, :] - X_tensor[:, -1, :]) ** 2
# # 假设我们要看第24000个点（异常点）
# for i in range(24000, 24000+100):
#     feature_errors = (X_reconstructed[i, -1, :] - X_tensor[i, -1, :]) ** 2
#     top_contributors = np.argsort(feature_errors.numpy())[::-1]
#     print(f"在点 {i} 处，重建误差最大的特征是: {[all_cols[i] for i in top_contributors[:3]]}")

# anomaly_idx = 24000
# top_contributors = np.argsort(feature_errors[anomaly_idx].numpy())[::-1]

# print(f"在点 {anomaly_idx} 处，重建误差最大的特征是: {[all_cols[i] for i in top_contributors[:3]]}")