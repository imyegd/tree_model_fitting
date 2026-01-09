import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 设置字体回退，让 matplotlib 自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

split_data_file = "./data/processed/split_data.npz"
model_file = "./models/pls_model.pkl"

if not os.path.exists(split_data_file):
    raise FileNotFoundError(f"未找到数据文件: {split_data_file}")

if not os.path.exists(model_file):
    raise FileNotFoundError(f"未找到已训练的模型文件: {model_file}")

print(f"加载测试数据: {split_data_file}")
data = np.load(split_data_file, allow_pickle=True)
# X_test = data['X_test']
# y_test = data['y_test']
X_test = data['X_train'][10000:11000]
y_test = data['y_train'][10000:11000]

print(f"加载已保存的模型: {model_file}")
artifacts = joblib.load(model_file)
pls = artifacts['pls_model']
scaler_X = artifacts['scaler_X']
scaler_Y = artifacts.get('scaler_Y', None)
eigen_values_pls = artifacts['eigen_values_pls']
UCL_T2X = artifacts['UCL_T2X']
UCL_SPEX = artifacts['UCL_SPEX_approx']
feature_names = artifacts.get('feature_names', [f'feature{i+1}' for i in range(X_test.shape[1])])

df_test_X = pd.DataFrame(X_test, columns=feature_names)
df_test_Y = pd.DataFrame(y_test, columns=['target'])

print("\n--- 步骤 I: 测试数据标准化 ---")
X_test_scaled = scaler_X.transform(df_test_X)
if scaler_Y is not None:
    Y_test_scaled = scaler_Y.transform(df_test_Y)
else:
    Y_test_scaled = df_test_Y.values

def compute_pls_stats(X_scaled, pls_model, eigen_values):
    """计算 $T^2_X$ 和 $SPE_X$ 统计量"""
    T = pls_model.transform(X_scaled)
    T2X = np.sum(T**2 / eigen_values, axis=1)
    P_X_matrix = pls_model.x_loadings_
    X_hat = np.dot(T, P_X_matrix.T)
    E_X = X_scaled - X_hat
    SPE_X = np.sum(E_X**2, axis=1)
    return T2X, SPE_X, E_X

print("\n--- 步骤 II: 计算测试集统计量 ---")
T2X_test, SPEX_test, E_X_test = compute_pls_stats(X_test_scaled, pls, eigen_values_pls)

df_monitor = pd.DataFrame({
    'T2X': T2X_test,
    'SPEX': SPEX_test,
    'T2X_Anomaly': T2X_test > UCL_T2X,
    'SPEX_Anomaly': SPEX_test > UCL_SPEX
})

first_anomaly_index = df_monitor[df_monitor['T2X_Anomaly'] | df_monitor['SPEX_Anomaly']].index.min()
if pd.notna(first_anomaly_index):
    print(f"\n第一个异常点出现在测试集索引 {first_anomaly_index}")

    if df_monitor.loc[first_anomaly_index, 'T2X_Anomaly']:
        t_anomaly = pls.transform(X_test_scaled[first_anomaly_index].reshape(1, -1))[0]
        T2X_contributions = np.sum((t_anomaly[:, np.newaxis] * pls.x_loadings_.T)**2 / eigen_values_pls[:, np.newaxis], axis=0)
        df_T2X_contrib = pd.Series(T2X_contributions, index=feature_names).sort_values(ascending=False)
        print("\n$T^2_X$ 异常 (幅度): 最可能导致漂移的特征 (Top 3):")
        print(df_T2X_contrib.head(3).to_string())

    if df_monitor.loc[first_anomaly_index, 'SPEX_Anomaly']:
        e_anomaly = E_X_test[first_anomaly_index]
        SPEX_contributions = e_anomaly**2
        df_SPEX_contrib = pd.Series(SPEX_contributions, index=feature_names).sort_values(ascending=False)
        print("\n$SPE_X$ 异常 (结构): 最可能导致结构变化的特征 (Top 3):")
        print(df_SPEX_contrib.head(3).to_string())
else:
    print("\n测试集中未检测到 $T^2_X$ 或 $SPE_X$ 异常。")

print("\n--- 步骤 III: 绘制测试集控制图 ---")
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
sample_index = np.arange(len(T2X_test))

ax[0].plot(sample_index, T2X_test, label='Test $T^2_X$', color='red')
ax[0].axhline(UCL_T2X, color='red', linestyle='--', label='UCL $T^2_X$')
ax[0].set_title('测试集 $T^2_X$ 控制图')
ax[0].set_ylabel('$T^2_X$ Value')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(sample_index, SPEX_test, label='Test $SPE_X$', color='red')
ax[1].axhline(UCL_SPEX, color='red', linestyle='--', label='UCL $SPE_X$')
ax[1].set_title('测试集 $SPE_X$ 控制图')
ax[1].set_ylabel('$SPE_X$ Value')
ax[1].set_xlabel('测试样本序号')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

