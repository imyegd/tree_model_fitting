import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
import json
import os
from datetime import datetime

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

# ========== 保存模型 ==========
model_save_dir = './result/anomaly_detection_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    print(f"创建模型保存目录: {model_save_dir}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存PyTorch模型
model_path = os.path.join(model_save_dir, f'lstm_vae_model_{timestamp}.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': {
        'n_features': 36,
        'h_dim': 64,
        'z_dim': 16
    },
    'seq_length': SEQ_LEN
}, model_path)
print(f"✓ LSTM-VAE模型已保存: {model_path}")

# 保存scaler
scaler_path = os.path.join(model_save_dir, f'lstm_vae_scaler_{timestamp}.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler已保存: {scaler_path}")

# 保存模型配置信息
config = {
    'model_type': 'LSTM_VAE',
    'model_name': 'Anomaly_Detection_LSTM_VAE',
    'timestamp': timestamp,
    'parameters': {
        'n_features': 36,
        'h_dim': 64,
        'z_dim': 16,
        'seq_length': SEQ_LEN,
        'train_size': train_size,
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001,
        'KLD_weight': 0.001
    },
    'features': all_cols,
    'detection_method': 'reconstruction_error',
    'threshold_method': '8-sigma (mean + 8*std)'
}

config_path = os.path.join(model_save_dir, f'lstm_vae_config_{timestamp}.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"✓ 配置已保存: {config_path}")
# ==============================

# 5. 异常检测与可视化
model.eval()
with torch.no_grad():
    recon_x_all, _, _ = model(X_tensor)
    # 计算综合重建误差
    vae_mse = torch.mean((recon_x_all[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()

# 设定阈值
threshold = np.mean(vae_mse[:train_size]) + 8 * np.std(vae_mse[:train_size])
is_anomaly = (vae_mse > threshold).astype(int)

# 保存阈值信息
threshold_info = {
    'threshold': float(threshold),
    'mean': float(np.mean(vae_mse[:train_size])),
    'std': float(np.std(vae_mse[:train_size])),
    'n_sigma': 8
}
threshold_path = os.path.join(model_save_dir, f'lstm_vae_threshold_{timestamp}.json')
with open(threshold_path, 'w', encoding='utf-8') as f:
    json.dump(threshold_info, f, indent=4)
print(f"✓ 阈值信息已保存: {threshold_path}")

# 提取target的原始值和重建值（target是第一个特征）
target_actual = scaler.inverse_transform(X_tensor[:, -1, :].numpy())[:, 0]  # 第0列是target
target_reconstructed = scaler.inverse_transform(recon_x_all[:, -1, :].numpy())[:, 0]

# 使用数值索引代替时间轴（避免datetime处理慢的问题）
index_axis = np.arange(len(target_actual))

# 创建可视化
plt.figure(figsize=(15, 10))

# 子图1：原始值 vs 重建值，标记异常点
plt.subplot(2, 1, 1)
plt.plot(index_axis, target_actual, label='Actual Target', alpha=0.7, linewidth=0.5)
plt.plot(index_axis, target_reconstructed, label='VAE Reconstructed', alpha=0.5, linestyle='--', linewidth=0.5)
# 只标记异常点
anomaly_indices = index_axis[is_anomaly == 1]
plt.scatter(anomaly_indices, target_actual[is_anomaly == 1], 
            color='red', s=10, label='Anomaly Detected', zorder=5)
plt.title('LSTM-VAE Anomaly Detection (Reconstruction Method)')
plt.xlabel('Time Index')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：重建误差曲线与阈值线
plt.subplot(2, 1, 2)
plt.plot(index_axis, vae_mse, label='VAE Reconstruction Error', color='purple', linewidth=0.5)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.6f})')
plt.title('VAE Reconstruction Error and Threshold')
plt.xlabel('Time Index')
plt.ylabel('MSE Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=1.0)
plt.savefig('./result/anoma_detection/LSTM_VAE.png')
plt.show()

print(f"检测完成！共发现 {is_anomaly.sum()} 个异常点。")
print(f"判定阈值为: {threshold:.6f}")


# =============================================================
# LSTM-VAE 异常诊断：针对 25000-25500 区间
# =============================================================

target_start, target_end = 25000, 25500

# 1. 提取该区间的重建误差详情
# recon_x_all 和 X_tensor 的形状均为 (N, SEQ_LEN, 36)
# 我们取最后一个时间步进行计算
detailed_errors = (recon_x_all[target_start:target_end, -1, :] - X_tensor[target_start:target_end, -1, :])**2
detailed_errors_np = detailed_errors.numpy()

# 2. 绘制特征贡献热力图 (Heatmap)
import seaborn as sns

plt.figure(figsize=(16, 10))
# 行代表特征名称，列代表相对时间步
ax = sns.heatmap(detailed_errors_np.T, 
                 yticklabels=all_cols, 
                 cmap='magma', # VAE 常用深色系调色盘
                 cbar_kws={'label': 'Reconstruction MSE'})

plt.title(f'LSTM-VAE Root Cause Analysis: Feature Contribution (Indices {target_start}-{target_end})', fontsize=15)
plt.xlabel('Time Step (Relative)', fontsize=12)
plt.ylabel('Features', fontsize=12)

# 优化横坐标刻度
ticks = np.arange(0, target_end - target_start, 50)
labels = np.arange(target_start, target_end, 50)
plt.xticks(ticks, labels)

plt.savefig('./result/anoma_detection/LSTM_VAE_Diagnosis_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 统计该段内平均贡献最大的前 5 个特征
mean_errors = np.mean(detailed_errors_np, axis=0)
top_5_idx = np.argsort(mean_errors)[::-1][:5]

print(f"\n--- LSTM-VAE 异常诊断报告 (区间: {target_start}-{target_end}) ---")
print(f"{'排名':<6} {'特征名称':<15} {'平均重建误差':<15}")
for i, idx in enumerate(top_5_idx):
    print(f"{i+1:<6} {all_cols[idx]:<15} {mean_errors[idx]:.8f}")

# 4. 寻找该区间内“最孤立”的异常点进行切片分析
max_error_time_rel = np.argmax(np.mean(detailed_errors_np, axis=1))
global_max_time = target_start + max_error_time_rel
point_errors = detailed_errors_np[max_error_time_rel]
top_point_features = np.argsort(point_errors)[::-1][:3]

print(f"\n区间内重建误差峰值时刻: {global_max_time}")
print(f"该时刻最可疑特征: {[all_cols[i] for i in top_point_features]}")