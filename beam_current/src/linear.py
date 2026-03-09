#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用线性回归拟合束流数据
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import json
from datetime import datetime
# 设置字体回退，让matplotlib自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def ensure_result_dir():
    """确保result目录存在"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/linear/{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    return result_dir

def load_and_prepare_data(csv_file_path, n_samples=None):
    """
    加载和准备数据
    
    Args:
        csv_file_path (str): CSV文件路径
        n_samples (int): 使用的样本数量，None表示使用所有数据
    
    Returns:
        tuple: (X, y, feature_columns) 特征矩阵、目标变量和特征列名
    """
    # 读取CSV文件
    print(f"正在读取数据文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 选择数据
    if n_samples is None:
        df_subset = df
        print(f"使用所有 {df.shape[0]} 条数据")
    else:
        df_subset = df.head(n_samples)
        print(f"使用前{n_samples}条数据")
    
    # 提取特征列（所有以feature开头的列）
    feature_columns = [col for col in df_subset.columns if col.startswith('feature')]
    print(f"特征列: {feature_columns}")
    
    # 提取目标变量
    if 'target' in df_subset.columns:
        target_column = 'target'
    elif '束流' in df_subset.columns:
        target_column = '束流'
    else:
        raise ValueError("未找到目标变量列")
    
    print(f"目标变量列: {target_column}")
    
    # 准备特征矩阵和目标变量
    X = df_subset[feature_columns].values
    y = df_subset[target_column].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    return X, y, feature_columns

def split_data(X, y, test_size=0.2, random_state=None, split_mode='time'):
    """
    划分训练集和测试集
    
    Args:
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        test_size (float): 测试集比例
        random_state (int): 随机种子，仅在split_mode='random'时使用
        split_mode (str): 划分模式，'time'表示按时间顺序划分，'random'表示随机划分
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if split_mode == 'random':
        # 使用sklearn的train_test_split进行随机划分
        print(f"使用随机划分模式 (random_state={random_state})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    elif split_mode == 'time':
        # 按时间顺序划分：前(1-test_size)为训练集，后test_size为测试集
        print("使用时间顺序划分模式")
        n_samples = X.shape[0]
        split_idx = int(n_samples * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
    else:
        raise ValueError(f"不支持的划分模式: {split_mode}，请使用'time'或'random'")
    
    print(f"训练集大小: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"测试集大小: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"训练集特征维度: {X_train.shape[1]}")
    print(f"测试集特征维度: {X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """
    训练线性回归模型
    
    Args:
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
    
    Returns:
        LinearRegression: 训练好的模型
    """
    print("开始训练线性回归模型...")
    
    # 创建线性回归模型
    model = LinearRegression()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("模型训练完成")
    print(f"模型系数: {model.coef_}")
    print(f"模型截距: {model.intercept_}")
    
    return model

def evaluate_model(model, X, y, dataset_name="数据集"):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 真实目标变量
        dataset_name (str): 数据集名称
    
    Returns:
        dict: 评估指标字典
    """
    # 预测
    y_pred = model.predict(X)
    
    # 计算评估指标
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"\n=== {dataset_name}评估结果 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    return metrics, y_pred

def save_model_and_results(model, train_metrics, test_metrics, feature_columns, 
                          y_train, y_train_pred, y_test, y_test_pred, result_dir,
                          split_mode='time', test_size=0.2, random_state=None):
    """
    保存模型和结果
    
    Args:
        model: 训练好的模型
        train_metrics (dict): 训练集评估指标
        test_metrics (dict): 测试集评估指标
        feature_columns (list): 特征列名
        y_train, y_train_pred: 训练集真实值和预测值
        y_test, y_test_pred: 测试集真实值和预测值
        result_dir (str): 结果保存目录
        split_mode (str): 数据划分模式
        test_size (float): 测试集比例
        random_state (int): 随机种子
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存模型
    model_path = os.path.join(result_dir, f"linear_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 2. 保存特征重要性CSV
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    importance_path = os.path.join(result_dir, f"linear_feature_importance_{timestamp}.csv")
    feature_importance_df.to_csv(importance_path, index=False, encoding='utf-8')
    print(f"特征重要性已保存到: {importance_path}")
    
    # 3. 保存评估指标到TXT文件
    metrics_path = os.path.join(result_dir, f"linear_metrics_{timestamp}.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("线性回归模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("数据划分配置:\n")
        f.write("-" * 50 + "\n")
        f.write(f"划分模式: {split_mode}\n")
        f.write(f"测试集比例: {test_size}\n")
        if split_mode == 'random':
            f.write(f"随机种子: {random_state}\n")
        f.write("\n")
        
        f.write("训练集评估结果:\n")
        f.write("-" * 50 + "\n")
        for metric_name, metric_value in train_metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")
        
        f.write("\n测试集评估结果:\n")
        f.write("-" * 50 + "\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"保存时间: {timestamp}\n")
        f.write("=" * 50 + "\n")
    
    print(f"评估指标已保存到: {metrics_path}")
    
    return {
        'model_path': model_path,
        'importance_path': importance_path,
        'metrics_path': metrics_path
    }

def plot_timeseries(y_train_true, y_train_pred, y_test_true, y_test_pred,
                    result_dir, title="线性回归 — 全局时序预测对比"):
    """
    绘制完整时序曲线图（单张独立大图）

    4 条线：
      蓝色实线   — 全部真实值（训练+测试连续）
      绿色虚线   — 训练集预测值（前 80%）
      红色虚线   — 测试集预测值（后 20%）
      灰色点线   — 训练/测试分界线

    Args:
        y_train_true  (numpy.ndarray): 训练集真实值
        y_train_pred  (numpy.ndarray): 训练集预测值
        y_test_true   (numpy.ndarray): 测试集真实值
        y_test_pred   (numpy.ndarray): 测试集预测值
        result_dir    (str): 结果保存目录
        title         (str): 图表标题
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_train = len(y_train_true)
    n_total = n_train + len(y_test_true)
    x_idx   = np.arange(n_total)

    all_true       = np.concatenate([y_train_true, y_test_true])
    train_pred_full = np.full(n_total, np.nan)
    test_pred_full  = np.full(n_total, np.nan)
    train_pred_full[:n_train] = y_train_pred
    test_pred_full[n_train:]  = y_test_pred

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(x_idx, all_true,
            color='#2563eb', linewidth=0.9, alpha=0.9,
            label='真实值（全部）')
    ax.plot(x_idx, train_pred_full,
            color='#16a34a', linewidth=0.9, linestyle='--', alpha=0.9,
            label='训练集预测值')
    ax.plot(x_idx, test_pred_full,
            color='#dc2626', linewidth=0.9, linestyle='--', alpha=0.9,
            label='测试集预测值')
    ax.axvline(x=n_train,
               color='gray', linewidth=1.4, linestyle=':',
               label=f'训练/测试分界 (n={n_train})')

    ax.set_xlabel('样本索引', fontsize=12)
    ax.set_ylabel('束流 (target)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(result_dir, f"linear_timeseries_{timestamp}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"时序曲线图已保存为: {plot_path}")

    plt.show()
    return plot_path


def plot_results(y_train_true, y_train_pred, y_test_true, y_test_pred,
                 feature_columns, model, result_dir, title="线性回归模型在测试集上的预测情况"):
    """
    三图布局（仿图2.9风格）：
      (a) 顶部全宽 — 测试集时序：真实值（蓝圆点）vs 预测值（红虚线星号）
      (b) 左下     — 散点图：True Values vs Predicted Values + Perfect Fit + R²
      (c) 右下     — 残差直方图 + KDE + STD 标注
    """
    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1],
                   hspace=0.42, wspace=0.32)

    x_idx = np.arange(len(y_test_true))

    # ── (a) 时序图 ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(x_idx, y_test_true,
              color='#2563eb', linewidth=1.2, alpha=0.85,
              label='True Values')
    ax_a.plot(x_idx, y_test_pred,
              color='#dc2626', linewidth=1.2, linestyle='--', alpha=0.85,
              label='Predicted Values')
    ax_a.set_xlabel('Sample Number', fontsize=11)
    ax_a.set_ylabel('束流 (target)', fontsize=11)
    ax_a.legend(loc='upper right', fontsize=10)
    ax_a.grid(True, alpha=0.25)
    ax_a.annotate('(a)', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=11)

    # ── (b) 散点图 ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.scatter(y_test_true, y_test_pred,
                 color='#2563eb', alpha=0.55, s=18, zorder=3)
    v_min = min(y_test_true.min(), y_test_pred.min())
    v_max = max(y_test_true.max(), y_test_pred.max())
    ax_b.plot([v_min, v_max], [v_min, v_max],
              color='#dc2626', linestyle='--', linewidth=1.8, label='Perfect Fit')
    r2 = r2_score(y_test_true, y_test_pred)
    ax_b.text(0.55, 0.12, f'$R^2 = {r2:.4f}$',
              transform=ax_b.transAxes, fontsize=12,
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    ax_b.set_xlabel('True Values', fontsize=11)
    ax_b.set_ylabel('Predicted Values', fontsize=11)
    ax_b.legend(fontsize=10, loc='upper left')
    ax_b.grid(True, alpha=0.25)
    ax_b.annotate('(b)', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=11)

    # ── (c) 残差直方图 + KDE ──────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    residuals = y_test_true - y_test_pred
    std_val   = residuals.std()
    bins_r    = np.histogram_bin_edges(residuals, bins=35)
    ax_c.hist(residuals, bins=bins_r, density=True,
              color='#93c5fd', edgecolor='white', linewidth=0.4, alpha=0.85)
    kde   = gaussian_kde(residuals, bw_method='scott')
    x_kde = np.linspace(residuals.min(), residuals.max(), 400)
    ax_c.plot(x_kde, kde(x_kde),
              color='#dc2626', linewidth=2.0)
    ax_c.text(0.97, 0.93, f'STD = {std_val:.4f}',
              transform=ax_c.transAxes, fontsize=11, ha='right',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    ax_c.set_xlabel('Residue (target)', fontsize=11)
    ax_c.set_ylabel('Frequency', fontsize=11)
    ax_c.grid(True, alpha=0.25)
    ax_c.annotate('(c)', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=11)

    plt.suptitle(title, fontsize=13, y=1.01)

    plot_path = os.path.join(result_dir, f"linear_regression_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"分析图已保存为: {plot_path}")

    plt.show()
    return plot_path

def main():
    # ==================== 配置参数 ====================
    # 数据划分模式: 'time' 表示按时间顺序划分, 'random' 表示随机划分
    SPLIT_MODE = 'time'  # 可选值: 'time' 或 'random'
    TEST_SIZE = 0.2      # 测试集比例
    RANDOM_STATE = 42    # 随机种子（仅在SPLIT_MODE='random'时使用）
    # ================================================
    
    # 确保结果目录存在
    result_dir = ensure_result_dir()
    
    # CSV文件路径
    csv_file = "data/raw/beamdata.csv"

    
    print(f"{'='*60}")
    print(f"数据划分模式: {SPLIT_MODE}")
    print(f"测试集比例: {TEST_SIZE}")
    if SPLIT_MODE == 'random':
        print(f"随机种子: {RANDOM_STATE}")
    print(f"{'='*60}\n")
    
    try:

        # 首次运行：加载原始数据并划分
        print("首次运行，加载原始数据...")
        X, y, feature_columns = load_and_prepare_data(csv_file, n_samples=None)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, split_mode=SPLIT_MODE
        )
        
        # 对特征进行标准化
        print("对特征进行标准化...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"标准化完成 - 均值: {scaler.mean_[:5]}... (显示前5个)")
        print(f"标准化完成 - 标准差: {scaler.scale_[:5]}... (显示前5个)")
            

        # 训练线性回归模型
        model = train_linear_regression(X_train, y_train)
        
        # 评估训练集性能
        train_metrics, y_train_pred = evaluate_model(model, X_train, y_train, "训练集")
        
        # 评估测试集性能
        test_metrics, y_test_pred = evaluate_model(model, X_test, y_test, "测试集")
        
        # 打印特征重要性（系数绝对值）
        print("\n=== 特征重要性（按系数绝对值排序）===")
        feature_importance = list(zip(feature_columns, np.abs(model.coef_)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):  # 显示前10个最重要的特征
            print(f"{i+1:2d}. {feature}: {importance:.6f}")
        
        # 保存模型和结果
        saved_paths = save_model_and_results(
            model, train_metrics, test_metrics, feature_columns,
            y_train, y_train_pred, y_test, y_test_pred, result_dir,
            split_mode=SPLIT_MODE, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # 绘制综合分析图
        plot_path = plot_results(
            y_train, y_train_pred, y_test, y_test_pred,
            feature_columns, model, result_dir
        )
        
        print(f"\n=== 所有结果已保存到 {result_dir} ===")
        print("保存的文件:")
        for key, path in saved_paths.items():
            print(f"  {key}: {os.path.basename(path)}")
        print(f"  分析图: {os.path.basename(plot_path)}")
        
        print("\n线性回归分析完成！")
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()