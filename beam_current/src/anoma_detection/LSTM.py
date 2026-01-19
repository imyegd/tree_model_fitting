import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import os
from datetime import datetime

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

# ========== 保存模型 ==========
model_save_dir = './result/anomaly_detection_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    print(f"创建模型保存目录: {model_save_dir}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存PyTorch模型
model_path = os.path.join(model_save_dir, f'lstm_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': {
        'input_dim': 35,
        'hidden_dim': 64,
        'num_layers': 2,
        'output_dim': 1
    },
    'seq_length': SEQ_LENGTH
}, model_path)
print(f"✓ LSTM模型已保存: {model_path}")

# 保存scalers
scaler_x_path = os.path.join(model_save_dir, f'lstm_scaler_x.pkl')
scaler_y_path = os.path.join(model_save_dir, f'lstm_scaler_y.pkl')
joblib.dump(scaler_x, scaler_x_path)
joblib.dump(scaler_y, scaler_y_path)
print(f"✓ Scaler_X已保存: {scaler_x_path}")
print(f"✓ Scaler_Y已保存: {scaler_y_path}")

# 保存模型配置信息
config = {
    'model_type': 'LSTM_Regressor',
    'model_name': 'Anomaly_Detection_LSTM',
    'timestamp': timestamp,
    'parameters': {
        'input_dim': 35,
        'hidden_dim': 64,
        'num_layers': 2,
        'output_dim': 1,
        'seq_length': SEQ_LENGTH,
        'train_size': train_size,
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    'features': feature_cols,
    'target': target_col,
    'detection_method': 'residual_based',
    'threshold_method': '3-sigma (mean + 30*std)',
    'device': str(device)
}

config_path = os.path.join(model_save_dir, f'lstm_config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"✓ 配置已保存: {config_path}")
# ==============================

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
threshold = np.mean(train_res) + 30 * np.std(train_res)
is_anomaly = (residuals > threshold).astype(int)

# 保存阈值信息
threshold_info = {
    'threshold': float(threshold),
    'mean': float(np.mean(train_res)),
    'std': float(np.std(train_res)),
    'n_sigma': 30
}
threshold_path = os.path.join(model_save_dir, f'lstm_threshold.json')
with open(threshold_path, 'w', encoding='utf-8') as f:
    json.dump(threshold_info, f, indent=4)
print(f"✓ 阈值信息已保存: {threshold_path}")

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
plt.savefig('./result/anoma_detection/LSTM.png')
plt.show()

print(f"检测完成，阈值设定为: {threshold:.4f}")

import shap

# --- 异常诊断部分：针对 23000-23500 点 ---

print(f"\n正在针对 23000 到 23500 点进行 SHAP 诊断分析...")

# 1. 准备背景数据集 (Background Dataset)
# SHAP 需要一个参考分布，通常从训练集中随机抽取一部分
background = X_tensor[:100].to(device) 

# 2. 创建解释器
# 对于 PyTorch 模型，DeepExplainer 是最常用的
explainer = shap.GradientExplainer(model, background)

# 3. 提取目标分析区间 (索引需要考虑 SEQ_LENGTH 的偏移)
# 因为 X_tensor 的第 0 个元素对应原数据的第 50 个点 (SEQ_LENGTH)
start_idx = 23000 - SEQ_LENGTH
end_idx = 23050 - SEQ_LENGTH
test_samples = X_tensor[start_idx:end_idx].to(device)

# 4. 计算 SHAP 值
# shap_values 的形状会是 (样本数, 时间步, 特征数)
shap_values = explainer.shap_values(test_samples)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 5. 诊断可视化
# 重点：因为 LSTM 输入是三维的，我们通常观察“最后一个时间步”对当前预测的影响
# 或者是将时间步维度进行平均
last_step_shap = shap_values[:, -1, :] # 取窗口中最后一个时间步的贡献
last_step_data = test_samples[:, -1, :].cpu().numpy() # 对应的特征原始值

# 5.1 摘要图 (Summary Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(last_step_shap, last_step_data, feature_names=feature_cols, show=False)
plt.title(f"Feature Importance via SHAP (Points 23000-23500)")
plt.savefig('./result/anoma_detection/LSTM_SHAP_Summary.png')
plt.show()

# 5.2 局部热力图 (观察这 500 个点内特征贡献的演变)
plt.figure(figsize=(15, 8))
# 转换为 DataFrame 方便绘图
shap_df = pd.DataFrame(last_step_shap, columns=feature_cols)
import seaborn as sns
sns.heatmap(shap_df.T, cmap='RdBu_r', center=0)
plt.title(f"SHAP Force Heatmap (Time 23000-23500)")
plt.xlabel("Relative Time Step")
plt.ylabel("Features")
plt.savefig('./result/anoma_detection/LSTM_SHAP_Heatmap.png')
plt.show()

# 5.3 找出最可疑的特征
mean_importance = np.abs(last_step_shap).mean(axis=0)
top_feature_idx = np.argsort(mean_importance)[::-1][:5]
print(f"该时间段内对 Target 影响最大的前 5 个特征是:")
for i in top_feature_idx:
    print(f"- {feature_cols[i]}: 平均贡献权重 {mean_importance[i]:.6f}")

# --- 异常诊断部分结束 ---