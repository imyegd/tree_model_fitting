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

# ========== 保存模型 ==========
model_save_dir = './result/anomaly_detection_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    print(f"创建模型保存目录: {model_save_dir}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存PyTorch模型
model_path = os.path.join(model_save_dir, f'lstm_ae_model_{timestamp}.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': {
        'n_features': 36,
        'embedding_dim': 16
    },
    'seq_length': SEQ_LEN
}, model_path)
print(f"✓ LSTM-AE模型已保存: {model_path}")

# 保存scaler
scaler_path = os.path.join(model_save_dir, f'lstm_ae_scaler_{timestamp}.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler已保存: {scaler_path}")

# 保存模型配置信息
config = {
    'model_type': 'LSTM_AutoEncoder',
    'model_name': 'Anomaly_Detection_LSTM_AE',
    'timestamp': timestamp,
    'parameters': {
        'n_features': 36,
        'embedding_dim': 16,
        'seq_length': SEQ_LEN,
        'train_size': 15000,
        'epochs': 20,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    'features': all_cols,
    'detection_method': 'reconstruction_error',
    'threshold_method': '8-sigma (mean + 8*std)'
}

config_path = os.path.join(model_save_dir, f'lstm_ae_config_{timestamp}.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"✓ 配置已保存: {config_path}")
# ==============================

# 4. 异常检测 (计算重建误差)
# model.eval()
# with torch.no_grad():
#     X_reconstructed = model(X_tensor)
#     # 计算每个时间步的平均重建误差 (MSE)
#     # 我们只关注序列最后一个点的重建误差
#     diff = (X_reconstructed[:, -1, :] - X_tensor[:, -1, :]) ** 2
#     reconstruction_errors = torch.mean(diff, dim=1).numpy()

# 4. 异常检测与可视化
model.eval()
with torch.no_grad():
    X_reconstructed = model(X_tensor)
    # 计算所有特征的综合重建误差 (MSE)
    mse_errors = torch.mean((X_reconstructed[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()

# 自动计算一个统计阈值（均值 + 8倍标准差）
threshold = np.mean(mse_errors[:15000]) + 8 * np.std(mse_errors[:15000])
is_anomaly = (mse_errors > threshold).astype(int)

# 保存阈值信息
threshold_info = {
    'threshold': float(threshold),
    'mean': float(np.mean(mse_errors[:15000])),
    'std': float(np.std(mse_errors[:15000])),
    'n_sigma': 8
}
threshold_path = os.path.join(model_save_dir, f'lstm_ae_threshold_{timestamp}.json')
with open(threshold_path, 'w', encoding='utf-8') as f:
    json.dump(threshold_info, f, indent=4)
print(f"✓ 阈值信息已保存: {threshold_path}")

# 提取target的原始值和重建值（target是第一个特征）
target_actual = scaler.inverse_transform(X_tensor[:, -1, :].numpy())[:, 0]  # 第0列是target
target_reconstructed = scaler.inverse_transform(X_reconstructed[:, -1, :].numpy())[:, 0]

# 使用数值索引代替时间轴（避免datetime处理慢的问题）
index_axis = np.arange(len(target_actual))

# 创建可视化
plt.figure(figsize=(15, 10))

# 子图1：原始值 vs 重建值，标记异常点
plt.subplot(2, 1, 1)
plt.plot(index_axis, target_actual, label='Actual Target', alpha=0.7, linewidth=0.5)
plt.plot(index_axis, target_reconstructed, label='Reconstructed Target', alpha=0.5, linestyle='--', linewidth=0.5)
# 只标记异常点
anomaly_indices = index_axis[is_anomaly == 1]
plt.scatter(anomaly_indices, target_actual[is_anomaly == 1], 
            color='red', s=10, label='Anomaly Detected', zorder=5)
plt.title('LSTM-AE Anomaly Detection (Reconstruction Method)')
plt.xlabel('Time Index')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：重建误差曲线与阈值线
plt.subplot(2, 1, 2)
plt.plot(index_axis, mse_errors, label='Reconstruction Error', color='orange', linewidth=0.5)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.6f})')
plt.title('Reconstruction Error and Threshold')
plt.xlabel('Time Index')
plt.ylabel('MSE Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=1.0)
plt.savefig('./result/anoma_detection/LSTM_AE.png')
plt.show()

print(f"检测完成！共发现 {is_anomaly.sum()} 个异常点。")
print(f"判定阈值为: {threshold:.6f}")

# =============================================================
# 指定时间段（25000-25500）的异常深度诊断
# =============================================================

target_start, target_end = 25000, 25500

# 1. 提取该时间段的详细误差数据
# 注意：X_reconstructed 和 X_tensor 已经包含了 SEQ_LEN 的偏移
# 如果您要对应的原始 CSV 索引是 25000，这里需要根据数据对齐情况微调
detailed_errors = (X_reconstructed[target_start:target_end, -1, :] - X_tensor[target_start:target_end, -1, :])**2
detailed_errors_np = detailed_errors.numpy()

# 2. 绘制特征贡献热图
import seaborn as sns

plt.figure(figsize=(16, 10))
# 对转置后的数据画热力图，行是特征，列是时间步
ax = sns.heatmap(detailed_errors_np.T, 
                 yticklabels=all_cols, 
                 cmap='YlOrRd', 
                 cbar_kws={'label': 'Reconstruction Error (MSE)'})

plt.title(f'Root Cause Analysis: Feature Contribution Heatmap (Indices {target_start}-{target_end})', fontsize=15)
plt.xlabel('Relative Time Steps (within the segment)', fontsize=12)
plt.ylabel('Features', fontsize=12)

# 优化坐标轴显示：每隔50个点显示一个刻度
plt.xticks(np.arange(0, target_end-target_start, 50), np.arange(target_start, target_end, 50))

plt.savefig(f'./result/anoma_detection/LSTM_AE_Diagnosis_{target_start}_{target_end}.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 统计该段内的核心影响因子
# 计算该时段内各特征的平均重建误差并排序
mean_errors_by_feature = np.mean(detailed_errors_np, axis=0)
top_indices = np.argsort(mean_errors_by_feature)[::-1]

print(f"\n--- 时间段 {target_start} 到 {target_end} 诊断报告 ---")
print(f"{'排名':<6} {'特征名称':<15} {'平均重建误差':<15}")
for i, idx in enumerate(top_indices[:5]):  # 展示前5个
    print(f"{i+1:<6} {all_cols[idx]:<15} {mean_errors_by_feature[idx]:.8f}")

# 4. 找到该段内最严重的异常时刻点进行精准剖析
max_error_idx = np.argmax(np.mean(detailed_errors_np, axis=1))
global_idx = target_start + max_error_idx
point_errors = detailed_errors_np[max_error_idx]
top_point_indices = np.argsort(point_errors)[::-1]

print(f"\n该时段内最严重的异常点发生在全局索引: {global_idx}")
print(f"该点主要诱发特征: {[all_cols[i] for i in top_point_indices[:3]]}")