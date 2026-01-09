import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f, chi2

import matplotlib
import matplotlib.pyplot as plt
import os
# 设置字体回退，让matplotlib自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N_features = 35


split_data_file = "./data/processed/split_data.npz"
scaler_file = "./data/processed/scaler.pkl"

if os.path.exists(split_data_file) and os.path.exists(scaler_file):
    # 从文件加载已划分和标准化的数据
    print(f"从文件加载数据: {split_data_file}")
    data = np.load(split_data_file, allow_pickle=True)
    # X_train = data['X_train'][3000:3300]
    # X_test = data['X_train'][3100:3400]2
    X_train = data['X_train']
    Y_train = data['y_train']
    X_test = data['X_test']
    Y_test = data['y_test']

df_train = pd.DataFrame(X_train, columns=[f'feature{i+1}' for i in range(N_features)])
df_test = pd.DataFrame(X_test, columns=[f'feature{i+1}' for i in range(N_features)])



# =================================================================
# I. 步骤一：数据准备与预处理
# =================================================================

print("--- 步骤 I: 数据预处理 ---")

# 1. 初始化标准化器并拟合训练集
scaler = StandardScaler()
# 仅在训练集上拟合（计算均值和标准差）
X_train_scaled = scaler.fit_transform(df_train)
# 使用训练集的参数对测试集进行转换
X_test_scaled = scaler.transform(df_test)

N = X_train_scaled.shape[0]  # 训练样本数
P = X_train_scaled.shape[1]  # 特征数 (35)


# =================================================================
# II. 步骤二：建立 PCA 模型和确定主成分数量
# =================================================================

print("--- 步骤 II: 建立 PCA 模型 ---")

# 1. 运行 PCA
pca = PCA(n_components=P)
pca.fit(X_train_scaled)

# 特征值 (Eigenvalues, 解释方差)
eigen_values = pca.explained_variance_

# 2. 确定主成分数量 k (截断)
# 采用累计方差贡献率法，设定阈值 90%
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
variance_threshold = 0.90
k = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1

print(f"原始特征数 P = {P}")
print(f"累计方差贡献率达到 {variance_threshold*100}% 所需的主成分数 k = {k}")

# 载荷矩阵 P (Loadings)
P_matrix = pca.components_[:k].T  # 取前 k 个主成分的载荷向量


# =================================================================
# III. 步骤三：计算 $T^2$ 和 $SPE$ 控制限
# =================================================================

print("--- 步骤 III: 计算控制限 (UCL) ---")

alpha = 0.05 # 显著性水平 (1 - 置信度)

# --- T^2 统计量控制限 ---
# 使用 F 分布计算 UCL
UCL_T2 = (k * (N**2 - 1) / (N * (N - k))) * f.ppf(1 - alpha, k, N - k)
print(f"T^2 控制上限 (UCL_T2): {UCL_T2:.4f}")


# --- SPE/Q 统计量控制限 ---
# 使用 Chi-square (卡方) 近似
theta1 = np.sum(eigen_values[k:])
theta2 = np.sum(eigen_values[k:]**2)
theta3 = np.sum(eigen_values[k:]**3)

h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
c_alpha = chi2.ppf(1 - alpha, df=theta2**2 / theta3)
UCL_SPE = theta1 * (h0 * c_alpha / theta2 + 1 - h0)
print(f"SPE 控制上限 (UCL_SPE): {UCL_SPE:.4f}")


# =================================================================
# IV. 步骤四：监控和异常判断（计算 $T^2$ 和 $SPE$）
# =================================================================

# 定义计算函数
def compute_mspc_stats(X_scaled, P_matrix, eigen_values_k):
    """计算给定数据的 T2 和 SPE 统计量"""
    
    # 1. 计算主成分得分 (Scores, T)
    T = np.dot(X_scaled, P_matrix)
    
    # 2. 计算 T^2 统计量
    # T2 = sum(t_i^2 / lambda_i)
    T2 = np.sum(T**2 / eigen_values_k, axis=1)
    
    # 3. 计算预测值 (X_hat) 和残差 (E)
    X_hat = np.dot(T, P_matrix.T)
    E = X_scaled - X_hat  # 残差矩阵 E
    
    # 4. 计算 SPE 统计量
    # SPE = sum(e_i^2) = ||E||^2
    SPE = np.sum(E**2, axis=1)
    
    return T2, SPE, E


# 提取前 k 个主成分的特征值
eigen_values_k = eigen_values[:k]

# 监控训练集 (用于验证)
T2_train, SPE_train, _ = compute_mspc_stats(X_train_scaled, P_matrix, eigen_values_k)

# 监控测试集 (新数据)
T2_test, SPE_test, E_test = compute_mspc_stats(X_test_scaled, P_matrix, eigen_values_k)


# =================================================================
# V. 步骤五：异常判断和根因分析 (以测试集为例)
# =================================================================

print("\n--- 步骤 V: 异常判断与根因分析 ---")

# 合并监控结果
df_monitor = pd.DataFrame({
    'T2': T2_test,
    'SPE': SPE_test,
    'T2_Anomaly': T2_test > UCL_T2,
    'SPE_Anomaly': SPE_test > UCL_SPE
})

# 找出第一个出现异常的点
first_anomaly_index = df_monitor[df_monitor['T2_Anomaly'] | df_monitor['SPE_Anomaly']].index.min()
if first_anomaly_index is not np.nan:
    
    print(f"\n第一个异常点出现在测试集索引 {first_anomaly_index - 250} (原始数据索引 {first_anomaly_index})")
    
    # ----------------------------------------------------
    # 根因分析: 贡献图 (Contribution Plot)
    # ----------------------------------------------------
    
    anomaly_idx_in_test = first_anomaly_index - 250
    
    # A. T^2 贡献度分析 (针对 幅度异常)
    if df_monitor.loc[first_anomaly_index, 'T2_Anomaly']:
        
        # 获取该样本的主成分得分
        t_anomaly = np.dot(X_test_scaled[anomaly_idx_in_test], P_matrix)
        
        # 计算每个原始特征对 T^2 的贡献
        # T2贡献度 Contrib_j = sum_{i=1}^{k} (t_i * p_{ij})^2 / lambda_i  (简化版)
        T2_contributions = np.sum( (t_anomaly[:, np.newaxis] * P_matrix.T)**2 / eigen_values_k[:, np.newaxis], axis=0)
        
        df_T2_contrib = pd.Series(T2_contributions, index=df_train.columns).sort_values(ascending=False)
        print("\nT^2 异常 (幅度): 最可能导致漂移的特征 (T^2 贡献 Top 3):")
        print(df_T2_contrib.head(3).to_string())

    # B. SPE 贡献度分析 (针对 结构异常)
    if df_monitor.loc[first_anomaly_index, 'SPE_Anomaly']:
        
        # 获取该样本的残差
        e_anomaly = E_test[anomaly_idx_in_test]
        
        # 计算每个原始特征对 SPE 的贡献 (即残差平方)
        SPE_contributions = e_anomaly**2
        
        df_SPE_contrib = pd.Series(SPE_contributions, index=df_train.columns).sort_values(ascending=False)
        print("\nSPE 异常 (结构): 最可能导致结构变化的特征 (SPE 贡献 Top 3):")
        print(df_SPE_contrib.head(3).to_string())

else:
    print("\n测试集中未检测到 T^2 或 SPE 异常 (恭喜，过程稳定!)")


# =================================================================
# VI. 可视化 (可选)
# =================================================================
# 绘制 T2 和 SPE 控制图，以便直观查看

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# T^2 控制图
ax[0].plot(np.arange(N), T2_train, label='Train $T^2$', color='blue', alpha=0.5)
ax[0].plot(np.arange(N, N + len(T2_test)), T2_test, label='Test $T^2$', color='red')
ax[0].axhline(UCL_T2, color='red', linestyle='--', label='UCL')
ax[0].set_title('$T^2$ 统计量控制图')
ax[0].set_ylabel('$T^2$ Value')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# SPE 控制图
ax[1].plot(np.arange(N), SPE_train, label='Train SPE', color='blue', alpha=0.5)
ax[1].plot(np.arange(N, N + len(SPE_test)), SPE_test, label='Test SPE', color='red')
ax[1].axhline(UCL_SPE, color='red', linestyle='--', label='UCL')
ax[1].set_title('SPE 统计量控制图')
ax[1].set_ylabel('SPE Value')
ax[1].set_xlabel('样本序号')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()