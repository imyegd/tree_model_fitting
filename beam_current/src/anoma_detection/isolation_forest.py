import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

# 1. 数据准备
df = pd.read_csv('data/raw/束流.csv')
all_cols = ['target'] + [f'feature{i}' for i in range(1, 36)]
data = df[all_cols].values

# 孤立森林对量纲不敏感，但建议做标准化以保持诊断的一致性
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. 训练孤立森林
# n_estimators: 森林中树的数量
# contamination: 你预期异常数据的占比（如果不确定，设为 'auto'）
iso_forest = IsolationForest(n_estimators=100, 
                             contamination=0.01, 
                             random_state=42, 
                             n_jobs=-1)

print("正在训练孤立森林...")
# 判定异常点：-1 代表异常，1 代表正常
df['anomaly_label'] = iso_forest.fit_predict(data_scaled)
# 计算异常得分：得分越小（越负）说明越异常
df['anomaly_score'] = iso_forest.decision_function(data_scaled)

# ========== 保存模型 ==========
model_save_dir = './result/anomaly_detection_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
    print(f"创建模型保存目录: {model_save_dir}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存模型
model_path = os.path.join(model_save_dir, f'isolation_forest_model.pkl')
joblib.dump(iso_forest, model_path)
print(f"✓ 模型已保存: {model_path}")

# 保存scaler
scaler_path = os.path.join(model_save_dir, f'isolation_forest_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler已保存: {scaler_path}")

# 保存模型配置信息
config = {
    'model_type': 'IsolationForest',
    'model_name': 'Anomaly_Detection_IsolationForest',
    'timestamp': timestamp,
    'parameters': {
        'n_estimators': 100,
        'contamination': 0.01,
        'random_state': 42
    },
    'features': all_cols,
    'detection_method': 'isolation_based',
    'description': 'Anomaly score: lower (more negative) means more anomalous'
}

config_path = os.path.join(model_save_dir, f'isolation_forest_config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"✓ 配置已保存: {config_path}")
# ==============================

# 3. 可视化异常得分与原始曲线
plt.figure(figsize=(15, 7))

# 子图1：Target 曲线与标记出的异常点
plt.subplot(2, 1, 1)
plt.plot(df['target'], label='Target', alpha=0.6)
anomalies = df[df['anomaly_label'] == -1]
plt.scatter(anomalies.index, anomalies['target'], color='red', s=5, label='Isolated Points')
plt.title('Isolation Forest: Detected Anomalies on Target')
plt.legend()

# 子图2：异常得分曲线（注意：得分越低越异常）
plt.subplot(2, 1, 2)
plt.plot(-df['anomaly_score'], color='green', label='Anomaly Intensity (Inversed Score)')
plt.title('Anomaly Score (Higher means more isolated)')
plt.xlabel('Time Step')
plt.legend()

plt.tight_layout()
plt.savefig('./result/anoma_detection/isolation_forest.png')

plt.show()

# 4. 异常诊断：通过 DIFFI 或简单的特征偏离度
# 既然孤立森林本身不易直接输出特征贡献，我们用最直观的方法：
# 查看异常时刻，哪些特征偏离了它们的全局均值最多
def simple_diagnostic(idx):
    point = data_scaled[idx]
    # 计算当前点各特征相对于全局均值的偏离程度（Z-Score）
    deviations = np.abs(point) 
    top_indices = np.argsort(deviations)[::-1]
    return [all_cols[i] for i in top_indices[:5]]

# 打印 24081 点附近的诊断
print(f"在点 24081 处，最可疑的离群特征是: {simple_diagnostic(24081)}")
