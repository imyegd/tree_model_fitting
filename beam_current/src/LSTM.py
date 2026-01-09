#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LSTM模型拟合束流数据
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def ensure_result_dir():
    """确保result目录存在"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/lstm/{timestamp}"
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
    print(f"特征列数量: {len(feature_columns)}")
    
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

def reshape_for_lstm(X, num_original_features=44):
    """
    将特征矩阵重塑为LSTM输入格式
    
    Args:
        X (numpy.ndarray): 特征矩阵 [样本数, 总特征数]
        num_original_features (int): 单个时间步的特征数量
    
    Returns:
        numpy.ndarray: 重塑后的数据 [样本数, 时间步长, 特征数]
    """
    n_samples = X.shape[0]
    n_total_features = X.shape[1]
    time_steps = n_total_features // num_original_features
    
    if n_total_features % num_original_features != 0:
        raise ValueError(f"总特征数 {n_total_features} 不能被原始特征数 {num_original_features} 整除")
    
    X_reshaped = X.reshape(n_samples, time_steps, num_original_features)
    print(f"数据重塑: {X.shape} -> {X_reshaped.shape}")
    print(f"时间步长: {time_steps}, 每步特征数: {num_original_features}")
    
    return X_reshaped

class LSTMModel(nn.Module):
    """LSTM回归模型"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM输出
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        # lstm_out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.fc(lstm_out[:, -1, :])
        
        return out

def train_lstm(model, train_loader, criterion, optimizer, device, epochs=100, verbose=True):
    """
    训练LSTM模型
    
    Args:
        model: LSTM模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备(CPU/GPU)
        epochs: 训练轮数
        verbose: 是否打印训练过程
    
    Returns:
        list: 训练损失历史
    """
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return loss_history

def evaluate_lstm(model, data_loader, device):
    """
    评估LSTM模型
    
    Args:
        model: LSTM模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        tuple: (y_true, y_pred)
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            
            y_true_list.append(batch_y.cpu().numpy())
            y_pred_list.append(outputs.squeeze().cpu().numpy())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    return y_true, y_pred, metrics

def save_model_and_results(model, scaler, train_metrics, test_metrics, loss_history,
                          y_train, y_train_pred, y_test, y_test_pred, result_dir,
                          split_mode='time', test_size=0.2, random_state=None):
    """
    保存模型和结果
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        train_metrics (dict): 训练集评估指标
        test_metrics (dict): 测试集评估指标
        loss_history (list): 训练损失历史
        y_train, y_train_pred: 训练集真实值和预测值
        y_test, y_test_pred: 测试集真实值和预测值
        result_dir (str): 结果保存目录
        split_mode (str): 数据划分模式
        test_size (float): 测试集比例
        random_state (int): 随机种子
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存模型
    model_path = os.path.join(result_dir, f"lstm_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 2. 保存标准化器
    scaler_path = os.path.join(result_dir, f"lstm_scaler_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")
    
    # 3. 保存结果
    results = {
        'data_split_config': {
            'split_mode': split_mode,
            'test_size': test_size,
            'random_state': random_state if split_mode == 'random' else None
        },
        'model_info': {
            'model_type': 'LSTM',
            'timestamp': timestamp,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(result_dir, f"lstm_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {results_path}")
    
    # 4. 保存评估指标到TXT文件
    metrics_path = os.path.join(result_dir, f"lstm_metrics_{timestamp}.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("LSTM模型评估结果\n")
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
        'scaler_path': scaler_path,
        'results_path': results_path,
        'metrics_path': metrics_path
    }

def plot_results(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                loss_history, result_dir, title="LSTM模型拟合结果"):
    """
    绘制拟合结果图
    
    Args:
        y_train_true (numpy.ndarray): 训练集真实值
        y_train_pred (numpy.ndarray): 训练集预测值
        y_test_true (numpy.ndarray): 测试集真实值
        y_test_pred (numpy.ndarray): 测试集预测值
        loss_history (list): 训练损失历史
        result_dir (str): 结果保存目录
        title (str): 图表标题
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 训练集拟合结果
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
    
    # 2. 测试集拟合结果
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
    
    # 3. 残差图
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
    
    # 4. 训练损失曲线
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(loss_history, 'b-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('训练损失曲线')
    ax4.grid(True, alpha=0.3)
    
    # 5. 训练集值分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_train_true, bins=30, alpha=0.7, label='训练集真实值', color='blue')
    ax5.hist(y_train_pred, bins=30, alpha=0.7, label='训练集预测值', color='red')
    ax5.set_xlabel('值')
    ax5.set_ylabel('频次')
    ax5.set_title('训练集值分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 测试集值分布
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
    plot_path = os.path.join(result_dir, f"lstm_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"分析图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def main():
    # ==================== 配置参数 ====================
    # 数据划分模式: 'time' 表示按时间顺序划分, 'random' 表示随机划分
    SPLIT_MODE = 'random'  # 可选值: 'time' 或 'random'
    TEST_SIZE = 0.2      # 测试集比例
    RANDOM_STATE = 42    # 随机种子（仅在SPLIT_MODE='random'时使用）
    
    # LSTM模型参数
    NUM_ORIGINAL_FEATURES = 44  # 单个时间步的特征数量
    HIDDEN_SIZE = 64            # LSTM隐藏层大小
    NUM_LAYERS = 2              # LSTM层数
    DROPOUT = 0.2               # Dropout比例
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    # ================================================
    
    # 确保结果目录存在
    result_dir = ensure_result_dir()
    
    # 文件路径
    csv_file = "./data/raw/束流.csv"

    
    print(f"{'='*60}")
    print(f"数据划分模式: {SPLIT_MODE}")
    print(f"测试集比例: {TEST_SIZE}")
    if SPLIT_MODE == 'random':
        print(f"随机种子: {RANDOM_STATE}")
    print(f"{'='*60}\n")
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
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
    
        
        # 重塑数据为LSTM输入格式
        print("\n重塑数据为LSTM输入格式...")
        X_train_reshaped = reshape_for_lstm(X_train, NUM_ORIGINAL_FEATURES)
        X_test_reshaped = reshape_for_lstm(X_test, NUM_ORIGINAL_FEATURES)
        
        # 转换为PyTorch张量
        print("转换为PyTorch张量...")
        X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"批次大小: {BATCH_SIZE}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # 创建LSTM模型
        print("\n创建LSTM模型...")
        model = LSTMModel(
            input_size=NUM_ORIGINAL_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        print(f"模型结构:\n{model}")
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 训练模型
        print(f"\n开始训练LSTM模型 (Epochs: {EPOCHS})...")
        loss_history = train_lstm(
            model, train_loader, criterion, optimizer, device, 
            epochs=EPOCHS, verbose=True
        )
        
        print("\nLSTM模型训练完成！")
        
        # 评估模型
        print("\n评估训练集...")
        y_train_true, y_train_pred, train_metrics = evaluate_lstm(model, train_loader, device)
        print("=== 训练集评估结果 ===")
        for metric_name, metric_value in train_metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
        
        print("\n评估测试集...")
        y_test_true, y_test_pred, test_metrics = evaluate_lstm(model, test_loader, device)
        print("=== 测试集评估结果 ===")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
        
        # 保存模型和结果
        saved_paths = save_model_and_results(
            model, scaler, train_metrics, test_metrics, loss_history,
            y_train_true, y_train_pred, y_test_true, y_test_pred, result_dir,
            split_mode=SPLIT_MODE, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # 绘制结果图
        plot_path = plot_results(
            y_train_true, y_train_pred, y_test_true, y_test_pred,
            loss_history, result_dir
        )
        
        print(f"\n=== 所有结果已保存到 {result_dir} ===")
        print("保存的文件:")
        for key, path in saved_paths.items():
            print(f"  {key}: {os.path.basename(path)}")
        print(f"  分析图: {os.path.basename(plot_path)}")
        
        print("\nLSTM回归分析完成！")
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
