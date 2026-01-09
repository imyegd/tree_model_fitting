import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 数据准备 (保持不变)
df = pd.read_csv('data/raw/束流.csv')
all_cols = ['target'] + [f'feature{i}' for i in range(1, 36)]
data = df[all_cols].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=50):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

SEQ_LEN = 50
X_seq = create_sequences(data_scaled, SEQ_LEN)
X_tensor = torch.from_numpy(X_seq).float()

train_size = 15000
train_loader = DataLoader(TensorDataset(X_tensor[:train_size]), batch_size=64, shuffle=True)

# 2. 定义 LSTM-VAE 模型
class LSTM_VAE(nn.Module):
    def __init__(self, n_features, h_dim=64, z_dim=16):
        super(LSTM_VAE, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(n_features, h_dim, batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(z_dim, h_dim)
        self.decoder_lstm = nn.LSTM(h_dim, n_features, batch_first=True)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        _, (h, _) = self.encoder_lstm(x)
        h = h[-1] # 取最后层隐藏状态
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # Decoder 准备
        z_out = self.decoder_fc(z)
        # 将 z 映射回序列长度
        decode_input = z_out.repeat(SEQ_LEN, 1, 1).transpose(0, 1)
        
        recon_x, _ = self.decoder_lstm(decode_input)
        return recon_x, mu, logvar

# 3. 定义损失函数 (MSE 重建损失 + KL 散度)
def loss_function(recon_x, x, mu, logvar):
    # 重建损失
    KLD_weight = 0.001 # KL 散度权重，可以微调
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL 散度: 促使隐空间分布接近标准正态分布
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD_weight * kld_loss

# 4. 训练模型
model = LSTM_VAE(n_features=36, h_dim=64, z_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.6f}")

# 5. 异常检测与可视化
model.eval()
with torch.no_grad():
    recon_x_all, _, _ = model(X_tensor)
    # 计算综合重建误差
    vae_mse = torch.mean((recon_x_all[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()

# 绘制异常得分曲线
plt.figure(figsize=(15, 5))
plt.plot(vae_mse, color='purple', label='VAE Anomaly Score')
# 设定阈值
threshold = np.mean(vae_mse[:train_size]) + 3 * np.std(vae_mse[:train_size])
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomaly Detection using LSTM-VAE')
plt.legend()
plt.show()

# 6. 特征贡献热力图 (诊断)
start_idx, end_idx = 24000, 24150
detailed_errors = (recon_x_all[start_idx:end_idx, -1, :] - X_tensor[start_idx:end_idx, -1, :])**2
detailed_errors_np = detailed_errors.numpy()

import seaborn as sns
plt.figure(figsize=(15, 8))
sns.heatmap(detailed_errors_np.T, yticklabels=all_cols, cmap='magma')
plt.title(f'VAE Feature Contribution (From {start_idx} to {end_idx})')
plt.show()