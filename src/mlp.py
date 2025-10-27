#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用多层感知机(MLP)拟合束流数据
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置字体回退，让matplotlib自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def ensure_result_dir():
    """确保result目录存在"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/束流/mlp/{timestamp}"
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

def split_data(X, y, test_size=0.2, random_state=None):
    """
    划分训练集和测试集（按顺序划分，不使用随机）
    
    Args:
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        test_size (float): 测试集比例
        random_state: 该参数已废弃，保留仅为兼容性
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 计算分割点
    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - test_size))
    
    # 按顺序划分：前80%为训练集，后20%为测试集
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    print(f"训练集特征维度: {X_train.shape[1]}")
    print(f"测试集特征维度: {X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_mlp(X_train, y_train, hidden_layer_sizes=(100,), max_iter=500, learning_rate_init=0.001):
    """
    训练MLP回归模型
    
    Args:
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
        hidden_layer_sizes (tuple): 隐藏层大小
        max_iter (int): 最大迭代次数
        learning_rate_init (float): 初始学习率
    
    Returns:
        tuple: (MLPRegressor, StandardScaler) 训练好的模型和标准化器
    """
    print("开始训练MLP模型...")
    
    # 标准化特征（MLP对输入数据的尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 创建MLP回归模型
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2正则化参数
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,  # 从训练集中分出10%作为验证集
        n_iter_no_change=10,  # 验证分数不改善时提前停止
        random_state=42,
        verbose=True
    )
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    print("MLP模型训练完成")
    print(f"迭代次数: {model.n_iter_}")
    print(f"损失值: {model.loss_:.6f}")
    
    return model, scaler

def evaluate_model(model, scaler, X, y, dataset_name="数据集"):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 真实目标变量
        dataset_name (str): 数据集名称
    
    Returns:
        tuple: (metrics_dict, y_pred)
    """
    # 标准化特征
    X_scaled = scaler.transform(X)
    
    # 预测
    y_pred = model.predict(X_scaled)
    
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

def save_model_and_results(model, scaler, train_metrics, test_metrics, feature_columns, 
                          y_train, y_train_pred, y_test, y_test_pred, result_dir):
    """
    保存模型和结果
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        train_metrics (dict): 训练集评估指标
        test_metrics (dict): 测试集评估指标
        feature_columns (list): 特征列名
        y_train, y_train_pred: 训练集真实值和预测值
        y_test, y_test_pred: 测试集真实值和预测值
        result_dir (str): 结果保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存模型和标准化器
    model_path = os.path.join(result_dir, f"mlp_model_{timestamp}.pkl")
    scaler_path = os.path.join(result_dir, f"mlp_scaler_{timestamp}.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"模型已保存到: {model_path}")
    print(f"标准化器已保存到: {scaler_path}")
    
    # 2. 保存模型信息
    model_info = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'solver': model.solver,
        'alpha': model.alpha,
        'learning_rate_init': model.learning_rate_init,
        'n_iter': model.n_iter_,
        'loss': float(model.loss_)
    }
    
    info_path = os.path.join(result_dir, f"mlp_model_info_{timestamp}.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print(f"模型信息已保存到: {info_path}")
    
    # 3. 保存评估指标到TXT文件
    metrics_path = os.path.join(result_dir, f"mlp_metrics_{timestamp}.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("MLP回归模型评估结果\n")
        f.write("=" * 50 + "\n\n")
        
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
        'scaler_path': scaler_path,
        'info_path': info_path,
        'metrics_path': metrics_path
    }

def plot_results(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                model, result_dir, title="MLP回归拟合结果"):
    """
    绘制拟合结果图
    
    Args:
        y_train_true (numpy.ndarray): 训练集真实值
        y_train_pred (numpy.ndarray): 训练集预测值
        y_test_true (numpy.ndarray): 测试集真实值
        y_test_pred (numpy.ndarray): 测试集预测值
        model: 训练好的模型
        result_dir (str): 结果保存目录
        title (str): 图表标题
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 训练集和测试集拟合结果对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_train_true, y_train_pred, alpha=0.6, s=20, color='blue', label='训练集')
    min_val = min(y_train_true.min(), y_train_pred.min())
    max_val = max(y_train_true.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title('训练集拟合结果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test_true, y_test_pred, alpha=0.6, s=20, color='green', label='测试集')
    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('预测值')
    ax2.set_title('测试集拟合结果')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax3 = plt.subplot(2, 3, 3)
    train_residuals = y_train_true - y_train_pred
    test_residuals = y_test_true - y_test_pred
    ax3.scatter(y_train_pred, train_residuals, alpha=0.6, s=20, color='blue', label='训练集残差')
    ax3.scatter(y_test_pred, test_residuals, alpha=0.6, s=20, color='green', label='测试集残差')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('预测值')
    ax3.set_ylabel('残差')
    ax3.set_title('残差图')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. 训练损失曲线（如果有的话）
    ax4 = plt.subplot(2, 3, 4)
    if hasattr(model, 'loss_curve_') and model.loss_curve_ is not None:
        ax4.plot(model.loss_curve_, 'b-', linewidth=2)
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('损失值')
        ax4.set_title('训练损失曲线')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '无损失曲线数据', ha='center', va='center')
        ax4.set_title('训练损失曲线')
    
    # 4. 预测值分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_train_true, bins=30, alpha=0.7, label='训练集真实值', color='blue')
    ax5.hist(y_train_pred, bins=30, alpha=0.7, label='训练集预测值', color='red')
    ax5.set_xlabel('值')
    ax5.set_ylabel('频次')
    ax5.set_title('训练集值分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. 测试集预测值分布
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_test_true, bins=30, alpha=0.7, label='测试集真实值', color='green')
    ax6.hist(y_test_pred, bins=30, alpha=0.7, label='测试集预测值', color='orange')
    ax6.set_xlabel('值')
    ax6.set_ylabel('频次')
    ax6.set_title('测试集值分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(result_dir, f"mlp_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"分析图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def main():
    # 确保结果目录存在
    result_dir = ensure_result_dir()
    
    # CSV文件路径
    csv_file = "./data/束流.csv"
    
    try:
        # 加载和准备数据（使用所有数据）
        X, y, feature_columns = load_and_prepare_data(csv_file, n_samples=None)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # 训练MLP回归模型
        model, scaler = train_mlp(
            X_train, y_train, 
            hidden_layer_sizes=(100, 50),  # 两个隐藏层：100和50个神经元
            max_iter=1000,
            learning_rate_init=0.001
        )
        
        # 评估训练集性能
        train_metrics, y_train_pred = evaluate_model(model, scaler, X_train, y_train, "训练集")
        
        # 评估测试集性能
        test_metrics, y_test_pred = evaluate_model(model, scaler, X_test, y_test, "测试集")
        
        # 保存模型和结果
        saved_paths = save_model_and_results(
            model, scaler, train_metrics, test_metrics, feature_columns,
            y_train, y_train_pred, y_test, y_test_pred, result_dir
        )
        
        # 绘制结果图
        plot_path = plot_results(
            y_train, y_train_pred, y_test, y_test_pred,
            model, result_dir
        )
        
        print(f"\n=== 所有结果已保存到 {result_dir} ===")
        print("保存的文件:")
        for key, path in saved_paths.items():
            print(f"  {key}: {os.path.basename(path)}")
        print(f"  分析图: {os.path.basename(plot_path)}")
        
        print("\nMLP回归分析完成！")
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
