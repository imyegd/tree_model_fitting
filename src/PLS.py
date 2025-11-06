import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression # 导入 PLS
from scipy.stats import f, chi2

# 设置字体回退，让matplotlib自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N_features = 35 # X 特征数

# =================================================================
# 模拟数据加载（根据您的原始代码结构）
# 假设 split_data.npz 包含了 X_train, y_train, X_test, y_test
# =================================================================

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
    X_train = data['X_train'][2000:]
    Y_train = data['y_train'][2000:]
    X_test = data['X_test']
    Y_test = data['y_test']

df_train_X = pd.DataFrame(X_train, columns=[f'feature{i+1}' for i in range(N_features)])
df_test_X = pd.DataFrame(X_test, columns=[f'feature{i+1}' for i in range(N_features)])
df_train_Y = pd.DataFrame(Y_train, columns=[f'target'])
df_test_Y = pd.DataFrame(Y_test, columns=[f'target'])

# =================================================================
# I. 步骤一：数据准备与预处理
# =================================================================

print("\n--- 步骤 I: 数据预处理 (X 和 Y 均需标准化) ---")

# 1. 标准化 X (仅在训练集上拟合)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(df_train_X)
X_test_scaled = scaler_X.transform(df_test_X)

# 2. 标准化 Y (仅在训练集上拟合)
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(df_train_Y)
Y_test_scaled = scaler_Y.transform(df_test_Y)

N = X_train_scaled.shape[0]  # 训练样本数
P = X_train_scaled.shape[1]  # X 特征数
M = Y_train_scaled.shape[1]  # Y 响应数


# =================================================================
# II. 步骤二：建立 PLS 模型和确定潜变量数量
# =================================================================

print("\n--- 步骤 II: 建立 PLS 模型 ---")

# 1. 确定潜变量数量 k (通常通过交叉验证或经验法则)
# 经验法则或选择一个能解释大部分方差的值
# 这里我们简化，直接选择一个值 (例如 5)
k = 5
print(f"潜变量数量 k = {k}")

# 2. 运行 PLS (使用 N_components=k)
# PLSRegression 会对 X 和 Y 进行中心化和缩放（如果 scale=True，但我们已经手动标准化）
pls = PLSRegression(n_components=k, scale=False)
pls.fit(X_train_scaled, Y_train_scaled)

# 提取 PLS 载荷和潜变量得分
# T: X 空间得分 (Scores) - (N x k)
# P_X: X 载荷 (Loadings) - (P x k)
# W: X 权重 (Weights) - (P x k)
# Q: Y 载荷 (Y-Loadings) - (M x k)

T_train = pls.x_scores_ # 训练集 X 空间得分 (N x k)
P_X = pls.x_loadings_ # X 载荷矩阵 (P x k)

# 计算潜变量的解释方差 (用于 T^2 统计量)
# PLS 潜变量的解释方差通常用潜变量得分的方差来近似，
# 或者使用奇异值分解得到的奇异值。
# 这里使用潜变量得分的方差作为其重要性的度量 (近似特征值)
eigen_values_pls = np.var(T_train, axis=0) # (k,)

print(f"PLS 潜变量 (Scores) 方差 (近似特征值): {eigen_values_pls}")

# =================================================================
# III. 步骤三：计算 $T^2$ 和 $SPE$ 控制限
# =================================================================

print("\n--- 步骤 III: 计算控制限 (UCL) ---")

alpha = 0.05 # 显著性水平 (1 - 置信度)

# --- $T^2_X$ 统计量控制限 (针对 X 空间) ---
# 使用 F 分布计算 UCL
UCL_T2X = (k * (N**2 - 1) / (N * (N - k))) * f.ppf(1 - alpha, k, N - k)
print(f"$T^2_X$ 控制上限 (UCL_T2X): {UCL_T2X:.4f}")

# --- $SPE_X/Q$ 统计量控制限 (针对 X 空间残差) ---
# X 空间残差的特征值：$\lambda_i = 1$ (如果残差空间是标准正交的，但 PLS 的残差空间不是)
# 使用 Chi-square (卡方) 近似，但需要残差空间的特征值，这在 PLS 中计算复杂。
# 更简单的近似：假设残差服从 Chi-square 分布，自由度为 $P-k$。
df_res = P - k
UCL_SPEX_approx = chi2.ppf(1 - alpha, df=df_res)
print(f"SPE_X 控制上限 (UCL_SPEX_approx, df={df_res}): {UCL_SPEX_approx:.4f}")

# (注: 更精确的 SPE 控制限计算需要使用残差协方差矩阵的特征值，此处使用简化的 $P-k$ 自由度近似。)

# =================================================================
# IV. 步骤四：监控和异常判断（计算 $T^2_X$ 和 $SPE_X$）
# =================================================================

def compute_pls_stats(X_scaled, pls_model, eigen_values_pls):
    """计算给定数据的 $T^2_X$ 和 $SPE_X$ 统计量"""
    
    # 1. 计算 X 空间得分 (Scores, T)
    T = pls_model.transform(X_scaled) # (N_sample x k)
    
    # 2. 计算 $T^2_X$ 统计量
    # $T^2_X = \sum_{i=1}^{k} \frac{t_i^2}{\lambda_i}$
    # $\lambda_i$ 是潜变量的解释方差 (eigen_values_pls)
    T2X = np.sum(T**2 / eigen_values_pls, axis=1) # (N_sample,)
    
    # 3. 计算 X 预测值 ($\hat{X}$) 和 X 残差 ($E_X$)
    # $\hat{X} = T P_X^T$ (或者使用 PLS 模型的逆变换，但 PLS 通常没有标准逆变换)
    # PLS 的 $\hat{X}$ 公式涉及 $\hat{Y} = T Q^T$ 和 $\hat{X} = \hat{Y} Q (W^T P_X)^T$ 等，非常复杂。
    # 最简单的方法是利用 $X = T P_X^T + E_X$
    # 注意: $\hat{X}$ 需要 X 载荷 $P_X$
    P_X_matrix = pls_model.x_loadings_
    X_hat = np.dot(T, P_X_matrix.T) # (N_sample x P)
    E_X = X_scaled - X_hat # X 残差矩阵 $E_X$
    
    # 4. 计算 $SPE_X$ 统计量 (X 残差平方和)
    # $SPE_X = ||E_X||^2 = \sum_{j=1}^{P} e_{X,j}^2$
    SPE_X = np.sum(E_X**2, axis=1) # (N_sample,)
    
    return T2X, SPE_X, E_X

# 监控训练集 (用于验证)
T2X_train, SPEX_train, _ = compute_pls_stats(X_train_scaled, pls, eigen_values_pls)

# 监控测试集 (新数据)
T2X_test, SPEX_test, E_X_test = compute_pls_stats(X_test_scaled, pls, eigen_values_pls)


# =================================================================
# V. 步骤五：异常判断和根因分析 (以测试集为例)
# =================================================================

print("\n--- 步骤 V: 异常判断与根因分析 (X 空间) ---")

# 合并监控结果
df_monitor = pd.DataFrame({
    'T2X': T2X_test,
    'SPEX': SPEX_test,
    'T2X_Anomaly': T2X_test > UCL_T2X,
    'SPEX_Anomaly': SPEX_test > UCL_SPEX_approx
})

# 找出第一个出现异常的点
first_anomaly_index = df_monitor[df_monitor['T2X_Anomaly'] | df_monitor['SPEX_Anomaly']].index.min()
if first_anomaly_index is not np.nan:
    
    print(f"\n第一个异常点出现在测试集索引 {first_anomaly_index}")
    
    # ----------------------------------------------------
    # 根因分析: 贡献图 (Contribution Plot)
    # ----------------------------------------------------
    
    anomaly_idx_in_test = first_anomaly_index
    
    # A. $T^2_X$ 贡献度分析 (针对 幅度异常)
    if df_monitor.loc[first_anomaly_index, 'T2X_Anomaly']:
        
        # 获取该样本的潜变量得分
        t_anomaly = pls.transform(X_test_scaled[anomaly_idx_in_test].reshape(1, -1))[0] # (k,)
        
        # 计算每个原始特征对 $T^2_X$ 的贡献
        # $T^2_X$ 贡献度 Contrib_j = $\sum_{i=1}^{k} (t_i * p_{X,ij})^2 / \lambda_i$
        T2X_contributions = np.sum( (t_anomaly[:, np.newaxis] * P_X.T)**2 / eigen_values_pls[:, np.newaxis], axis=0)
        
        df_T2X_contrib = pd.Series(T2X_contributions, index=df_train_X.columns).sort_values(ascending=False)
        print("\n$T^2_X$ 异常 (幅度): 最可能导致漂移的特征 ($T^2_X$ 贡献 Top 3):")
        print(df_T2X_contrib.head(3).to_string())

    # B. $SPE_X$ 贡献度分析 (针对 结构异常)
    if df_monitor.loc[first_anomaly_index, 'SPEX_Anomaly']:
        
        # 获取该样本的 X 残差
        e_anomaly = E_X_test[anomaly_idx_in_test]
        
        # 计算每个原始特征对 $SPE_X$ 的贡献 (即残差平方)
        SPEX_contributions = e_anomaly**2
        
        df_SPEX_contrib = pd.Series(SPEX_contributions, index=df_train_X.columns).sort_values(ascending=False)
        print("\n$SPE_X$ 异常 (结构): 最可能导致结构变化的特征 ($SPE_X$ 贡献 Top 3):")
        print(df_SPEX_contrib.head(3).to_string())

else:
    print("\n测试集中未检测到 $T^2_X$ 或 $SPE_X$ 异常。")


# =================================================================
# VI. 可视化
# =================================================================

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# $T^2_X$ 控制图
ax[0].plot(np.arange(N), T2X_train, label='Train $T^2_X$', color='blue', alpha=0.5)
ax[0].plot(np.arange(N, N + len(T2X_test)), T2X_test, label='Test $T^2_X$', color='red')
ax[0].axhline(UCL_T2X, color='red', linestyle='--', label='UCL $T^2_X$')
ax[0].axvline(N, color='gray', linestyle='-', alpha=0.5)
ax[0].set_title('$T^2_X$ 统计量控制图 (X 空间幅度异常)')
ax[0].set_ylabel('$T^2_X$ Value')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# $SPE_X$ 控制图
ax[1].plot(np.arange(N), SPEX_train, label='Train $SPE_X$', color='blue', alpha=0.5)
ax[1].plot(np.arange(N, N + len(SPEX_test)), SPEX_test, label='Test $SPE_X$', color='red')
ax[1].axhline(UCL_SPEX_approx, color='red', linestyle='--', label='UCL $SPE_X$')
ax[1].axvline(N, color='gray', linestyle='-', alpha=0.5)
ax[1].set_title('$SPE_X$ 统计量控制图 (X 空间结构异常)')
ax[1].set_ylabel('$SPE_X$ Value')
ax[1].set_xlabel('样本序号')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()