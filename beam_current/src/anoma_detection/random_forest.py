import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. 加载数据 (假设文件名是 data.csv)
df = pd.read_csv('data/raw/束流.csv', parse_dates=['时间'])
features = [f'feature{i}' for i in range(1, 36)]
target = 'target'

# 2. 选取“基准训练集”
# 建议选取前面相对平稳的一段（比如前15000行）作为正常规律的学习区
train_size = 15000
train_df = df.iloc[:train_size]
X_train = train_df[features]
y_train = train_df[target]

# 3. 训练随机森林回归模型
print("正在训练基准模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 对全量数据进行预测并计算残差
print("正在计算全量数据残差...")
df['target_pred'] = model.predict(df[features])
df['residual'] = (df[target] - df['target_pred']).abs()

# 5. 基于残差分布设定异常阈值 (3-Sigma 原则)
# 我们只基于训练集（正常段）的残差来计算阈值，这样标准更纯净
train_residuals = df['residual'].iloc[:train_size]
mu = train_residuals.mean()
sigma = train_residuals.std()
threshold = mu + 30 * sigma
# 标记异常：1为异常，0为正常
df['is_anomaly'] = (df['residual'] > threshold).astype(int)


import shap

# --- 异常诊断部分开始 ---

# 1. 筛选出异常样本
anomalies = df[df['is_anomaly'] == 1]

if len(anomalies) > 0:
    print(f"正在对 {len(anomalies)} 个异常点进行 SHAP 归因诊断...")
    
    # 2. 创建 SHAP 解释器 (TreeExplainer 专门用于随机森林)
    explainer = shap.TreeExplainer(model)
    
    # 为了计算速度，如果异常点太多，可以采样前 100 个
    diagnosis_samples = anomalies[features].head(100) 
    shap_values = explainer.shap_values(diagnosis_samples)
    
    # 3. 可视化诊断结果
    # 摘要图：展示哪些变量对 target 偏离贡献最大
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, diagnosis_samples, plot_type="bar", show=False)
    plt.title("Feature Contribution to Anomalies (SHAP)")
    plt.savefig('./result/anoma_detection/rf_shap_diagnosis.png')
    plt.show()

    # 4. 打印单个具体异常点的诊断 (例如第一个异常点)
    print("\n第一个异常点的详细特征贡献排名:")
    # 计算单个样本的 shap 绝对值贡献
    sample_idx = 0
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values[sample_idx])
    }).sort_values(by='importance', ascending=False)
    
    print(feature_importance.head(5))

else:
    print("未发现异常点，跳过诊断。")

# --- 异常诊断部分结束 ---



# 6. 可视化结果
plt.figure(figsize=(15, 8))

# 子图1：原始值 vs 预测值
plt.subplot(2, 1, 1)
plt.plot(df['时间'], df[target], label='Actual Target', alpha=0.7)
plt.plot(df['时间'], df['target_pred'], label='Predicted (Baseline)', alpha=0.5, linestyle='--')
plt.scatter(df.loc[df['is_anomaly'] == 1, '时间'], 
            df.loc[df['is_anomaly'] == 1, target], 
            color='red', s=10, label='Anomaly detected')
plt.title('Target Anomaly Detection (Residual Method)')
plt.legend()

# 子图2：残差曲线与阈值线
plt.subplot(2, 1, 2)
plt.plot(df['时间'], df['residual'], label='Residual Error', color='orange')
plt.axhline(y=threshold, color='red', linestyle='--', label='3-Sigma Threshold')
plt.title('Residual Scores and Threshold')
plt.legend()

plt.tight_layout()
plt.savefig('./result/anoma_detection/random_forest.png')
plt.show()

print(f"检测完成！共发现 {df['is_anomaly'].sum()} 个异常点。")
print(f"判定阈值为: {threshold:.4f}")