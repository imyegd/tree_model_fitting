#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测模型测试脚本
使用训练好的模型对束流数据进行异常检测和诊断
"""

import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置区 =====================
# 选择要使用的模型类型
MODEL_TYPE = 'random_forest'  # 可选: 'random_forest', 'isolation_forest', 'lstm', 'lstm_ae', 'lstm_vae'

# 模型文件路径 (修改为你的模型文件名)
MODEL_DIR = './result/anomaly_detection_models'
MODEL_TIMESTAMP = '20260111_102753'  # 修改为你的模型时间戳

# 数据文件路径
DATA_FILE = 'data/raw/束流.csv'

# 要检测的数据范围 (None表示全量数据)
START_IDX = None  # 例如: 20000
END_IDX = None    # 例如: 30000

# 是否保存检测结果
SAVE_RESULTS = True
RESULT_DIR = './result/anoma_detection/test_results'
# ================================================


class LSTMRegressor(nn.Module):
    """LSTM回归模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTM_AE(nn.Module):
    """LSTM自编码器"""
    def __init__(self, n_features, embedding_dim, seq_len):
        super(LSTM_AE, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, embedding_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, n_features, num_layers=1, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        repeat_hidden = hidden.repeat(self.seq_len, 1, 1).transpose(0, 1)
        out, _ = self.decoder(repeat_hidden)
        return out


class LSTM_VAE(nn.Module):
    """LSTM变分自编码器"""
    def __init__(self, n_features, h_dim, z_dim, seq_len):
        super(LSTM_VAE, self).__init__()
        self.seq_len = seq_len
        
        self.encoder_lstm = nn.LSTM(n_features, h_dim, batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        
        self.decoder_fc = nn.Linear(z_dim, h_dim)
        self.decoder_lstm = nn.LSTM(h_dim, n_features, batch_first=True)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        h = h[-1]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        z_out = self.decoder_fc(z)
        decode_input = z_out.repeat(self.seq_len, 1, 1).transpose(0, 1)
        recon_x, _ = self.decoder_lstm(decode_input)
        return recon_x, mu, logvar


def load_model_and_config(model_type, timestamp):
    """加载模型和配置文件"""
    print(f"\n{'='*60}")
    print(f"加载 {model_type.upper()} 模型 (时间戳: {timestamp})")
    print(f"{'='*60}")
    
    # 加载配置
    config_path = os.path.join(MODEL_DIR, f'{model_type}_config_{timestamp}.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"✓ 配置加载成功")
    
    # 加载阈值信息
    threshold_path = os.path.join(MODEL_DIR, f'{model_type}_threshold_{timestamp}.json')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r', encoding='utf-8') as f:
            threshold_info = json.load(f)
        print(f"✓ 阈值信息加载成功: {threshold_info['threshold']:.6f}")
    else:
        threshold_info = None
    
    # 根据模型类型加载模型
    if model_type == 'random_forest':
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model_{timestamp}.pkl')
        model = joblib.load(model_path)
        scaler = None
        print(f"✓ RandomForest模型加载成功")
        
    elif model_type == 'isolation_forest':
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model_{timestamp}.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{model_type}_scaler_{timestamp}.pkl')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✓ IsolationForest模型加载成功")
        print(f"✓ Scaler加载成功")
        
    elif model_type == 'lstm':
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model_{timestamp}.pth')
        scaler_x_path = os.path.join(MODEL_DIR, f'{model_type}_scaler_x_{timestamp}.pkl')
        scaler_y_path = os.path.join(MODEL_DIR, f'{model_type}_scaler_y_{timestamp}.pkl')
        
        checkpoint = torch.load(model_path)
        arch = checkpoint['model_architecture']
        
        model = LSTMRegressor(
            input_dim=arch['input_dim'],
            hidden_dim=arch['hidden_dim'],
            num_layers=arch['num_layers'],
            output_dim=arch['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = {
            'scaler_x': joblib.load(scaler_x_path),
            'scaler_y': joblib.load(scaler_y_path),
            'seq_length': checkpoint['seq_length']
        }
        print(f"✓ LSTM模型加载成功")
        print(f"✓ Scalers加载成功")
        
    elif model_type == 'lstm_ae':
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model_{timestamp}.pth')
        scaler_path = os.path.join(MODEL_DIR, f'{model_type}_scaler_{timestamp}.pkl')
        
        checkpoint = torch.load(model_path)
        arch = checkpoint['model_architecture']
        
        model = LSTM_AE(
            n_features=arch['n_features'],
            embedding_dim=arch['embedding_dim'],
            seq_len=checkpoint['seq_length']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = {
            'scaler': joblib.load(scaler_path),
            'seq_length': checkpoint['seq_length']
        }
        print(f"✓ LSTM-AE模型加载成功")
        print(f"✓ Scaler加载成功")
        
    elif model_type == 'lstm_vae':
        model_path = os.path.join(MODEL_DIR, f'{model_type}_model_{timestamp}.pth')
        scaler_path = os.path.join(MODEL_DIR, f'{model_type}_scaler_{timestamp}.pkl')
        
        checkpoint = torch.load(model_path)
        arch = checkpoint['model_architecture']
        
        model = LSTM_VAE(
            n_features=arch['n_features'],
            h_dim=arch['h_dim'],
            z_dim=arch['z_dim'],
            seq_len=checkpoint['seq_length']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = {
            'scaler': joblib.load(scaler_path),
            'seq_length': checkpoint['seq_length']
        }
        print(f"✓ LSTM-VAE模型加载成功")
        print(f"✓ Scaler加载成功")
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model, scaler, config, threshold_info


def detect_anomalies_rf(model, df, config, threshold_info, start_idx, end_idx):
    """使用RandomForest进行异常检测"""
    features = config['features']
    target = config['target']
    
    # 选择数据范围
    if start_idx is not None and end_idx is not None:
        df_test = df.iloc[start_idx:end_idx].copy()
    else:
        df_test = df.copy()
    
    print(f"\n检测数据范围: {len(df_test)} 个样本")
    
    # 预测
    X_test = df_test[features]
    y_pred = model.predict(X_test)
    residuals = np.abs(df_test[target].values - y_pred)
    
    # 异常判定
    threshold = threshold_info['threshold']
    is_anomaly = (residuals > threshold).astype(int)
    
    anomaly_count = is_anomaly.sum()
    print(f"检测到 {anomaly_count} 个异常点 ({anomaly_count/len(df_test)*100:.2f}%)")
    
    return {
        'predictions': y_pred,
        'residuals': residuals,
        'is_anomaly': is_anomaly,
        'threshold': threshold,
        'actual': df_test[target].values,
        'time': df_test['时间'].values if '时间' in df_test.columns else np.arange(len(df_test))
    }


def detect_anomalies_if(model, scaler, df, config, start_idx, end_idx):
    """使用IsolationForest进行异常检测"""
    all_cols = config['features']
    
    # 选择数据范围
    if start_idx is not None and end_idx is not None:
        df_test = df.iloc[start_idx:end_idx].copy()
    else:
        df_test = df.copy()
    
    print(f"\n检测数据范围: {len(df_test)} 个样本")
    
    # 数据准备
    data = df_test[all_cols].values
    data_scaled = scaler.transform(data)
    
    # 异常检测
    anomaly_labels = model.predict(data_scaled)
    anomaly_scores = model.decision_function(data_scaled)
    
    is_anomaly = (anomaly_labels == -1).astype(int)
    anomaly_count = is_anomaly.sum()
    print(f"检测到 {anomaly_count} 个异常点 ({anomaly_count/len(df_test)*100:.2f}%)")
    
    return {
        'anomaly_labels': anomaly_labels,
        'anomaly_scores': anomaly_scores,
        'is_anomaly': is_anomaly,
        'actual': df_test['target'].values,
        'time': df_test['时间'].values if '时间' in df_test.columns else np.arange(len(df_test))
    }


def detect_anomalies_lstm(model, scaler, df, config, threshold_info, start_idx, end_idx):
    """使用LSTM进行异常检测"""
    feature_cols = config['features']
    target_col = config['target']
    seq_length = scaler['seq_length']
    
    # 数据准备
    X_scaled = scaler['scaler_x'].transform(df[feature_cols])
    y_scaled = scaler['scaler_y'].transform(df[[target_col]])
    
    # 构造序列
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # 选择数据范围
    if start_idx is not None and end_idx is not None:
        # 注意序列偏移
        start = max(0, start_idx - seq_length)
        end = min(len(X_seq), end_idx - seq_length)
        X_seq = X_seq[start:end]
        y_seq = y_seq[start:end]
    
    print(f"\n检测数据范围: {len(X_seq)} 个样本 (序列长度: {seq_length})")
    
    # 预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_tensor = torch.from_numpy(X_seq).float().to(device)
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()
    
    # 反归一化
    y_actual = scaler['scaler_y'].inverse_transform(y_seq).flatten()
    y_pred = scaler['scaler_y'].inverse_transform(predictions_scaled).flatten()
    
    # 计算残差和异常
    residuals = np.abs(y_actual - y_pred)
    threshold = threshold_info['threshold']
    is_anomaly = (residuals > threshold).astype(int)
    
    anomaly_count = is_anomaly.sum()
    print(f"检测到 {anomaly_count} 个异常点 ({anomaly_count/len(X_seq)*100:.2f}%)")
    
    return {
        'predictions': y_pred,
        'residuals': residuals,
        'is_anomaly': is_anomaly,
        'threshold': threshold,
        'actual': y_actual,
        'time': df['时间'].iloc[seq_length:seq_length+len(X_seq)].values if '时间' in df.columns else np.arange(len(X_seq))
    }


def detect_anomalies_lstm_ae(model, scaler, df, config, threshold_info, start_idx, end_idx):
    """使用LSTM-AE进行异常检测"""
    all_cols = config['features']
    seq_length = scaler['seq_length']
    
    # 数据准备
    data = df[all_cols].values
    data_scaled = scaler['scaler'].transform(data)
    
    # 构造序列
    def create_sequences(data, seq_len):
        seqs = []
        for i in range(len(data) - seq_len):
            seqs.append(data[i:i+seq_len])
        return np.array(seqs)
    
    X_seq = create_sequences(data_scaled, seq_length)
    
    # 选择数据范围
    if start_idx is not None and end_idx is not None:
        start = max(0, start_idx - seq_length)
        end = min(len(X_seq), end_idx - seq_length)
        X_seq = X_seq[start:end]
    
    print(f"\n检测数据范围: {len(X_seq)} 个样本 (序列长度: {seq_length})")
    
    # 预测
    X_tensor = torch.from_numpy(X_seq).float()
    with torch.no_grad():
        X_reconstructed = model(X_tensor)
    
    # 计算重建误差
    mse_errors = torch.mean((X_reconstructed[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()
    
    # 异常判定
    threshold = threshold_info['threshold']
    is_anomaly = (mse_errors > threshold).astype(int)
    
    # 提取target值
    target_actual = scaler['scaler'].inverse_transform(X_tensor[:, -1, :].numpy())[:, 0]
    target_reconstructed = scaler['scaler'].inverse_transform(X_reconstructed[:, -1, :].numpy())[:, 0]
    
    anomaly_count = is_anomaly.sum()
    print(f"检测到 {anomaly_count} 个异常点 ({anomaly_count/len(X_seq)*100:.2f}%)")
    
    return {
        'reconstruction_errors': mse_errors,
        'is_anomaly': is_anomaly,
        'threshold': threshold,
        'actual': target_actual,
        'reconstructed': target_reconstructed,
        'time': df['时间'].iloc[seq_length:seq_length+len(X_seq)].values if '时间' in df.columns else np.arange(len(X_seq))
    }


def detect_anomalies_lstm_vae(model, scaler, df, config, threshold_info, start_idx, end_idx):
    """使用LSTM-VAE进行异常检测"""
    all_cols = config['features']
    seq_length = scaler['seq_length']
    
    # 数据准备
    data = df[all_cols].values
    data_scaled = scaler['scaler'].transform(data)
    
    # 构造序列
    def create_sequences(data, seq_len):
        seqs = []
        for i in range(len(data) - seq_len):
            seqs.append(data[i:i+seq_len])
        return np.array(seqs)
    
    X_seq = create_sequences(data_scaled, seq_length)
    
    # 选择数据范围
    if start_idx is not None and end_idx is not None:
        start = max(0, start_idx - seq_length)
        end = min(len(X_seq), end_idx - seq_length)
        X_seq = X_seq[start:end]
    
    print(f"\n检测数据范围: {len(X_seq)} 个样本 (序列长度: {seq_length})")
    
    # 预测
    X_tensor = torch.from_numpy(X_seq).float()
    with torch.no_grad():
        recon_x_all, _, _ = model(X_tensor)
    
    # 计算重建误差
    vae_mse = torch.mean((recon_x_all[:, -1, :] - X_tensor[:, -1, :])**2, dim=1).numpy()
    
    # 异常判定
    threshold = threshold_info['threshold']
    is_anomaly = (vae_mse > threshold).astype(int)
    
    # 提取target值
    target_actual = scaler['scaler'].inverse_transform(X_tensor[:, -1, :].numpy())[:, 0]
    target_reconstructed = scaler['scaler'].inverse_transform(recon_x_all[:, -1, :].numpy())[:, 0]
    
    anomaly_count = is_anomaly.sum()
    print(f"检测到 {anomaly_count} 个异常点 ({anomaly_count/len(X_seq)*100:.2f}%)")
    
    return {
        'reconstruction_errors': vae_mse,
        'is_anomaly': is_anomaly,
        'threshold': threshold,
        'actual': target_actual,
        'reconstructed': target_reconstructed,
        'time': df['时间'].iloc[seq_length:seq_length+len(X_seq)].values if '时间' in df.columns else np.arange(len(X_seq))
    }


def visualize_results(results, model_type, save_path=None):
    """可视化检测结果"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    time_axis = results['time']
    is_anomaly = results['is_anomaly']
    actual = results['actual']
    
    # 子图1: 实际值和异常点
    ax1 = axes[0]
    ax1.plot(time_axis, actual, label='Actual Target', alpha=0.7, linewidth=0.8)
    
    if 'predictions' in results:
        ax1.plot(time_axis, results['predictions'], label='Predicted', alpha=0.5, linestyle='--', linewidth=0.8)
    elif 'reconstructed' in results:
        ax1.plot(time_axis, results['reconstructed'], label='Reconstructed', alpha=0.5, linestyle='--', linewidth=0.8)
    
    # 标记异常点
    anomaly_mask = is_anomaly == 1
    ax1.scatter(time_axis[anomaly_mask], actual[anomaly_mask], 
                color='red', s=15, label=f'Anomaly ({anomaly_mask.sum()} points)', zorder=5)
    
    ax1.set_title(f'{model_type.upper()} Anomaly Detection Results', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time / Index')
    ax1.set_ylabel('Target Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 误差/得分曲线
    ax2 = axes[1]
    
    if 'residuals' in results:
        ax2.plot(time_axis, results['residuals'], label='Residual Error', color='orange', linewidth=0.8)
        if 'threshold' in results:
            ax2.axhline(y=results['threshold'], color='red', linestyle='--', label='Threshold', linewidth=2)
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual Error')
    elif 'reconstruction_errors' in results:
        ax2.plot(time_axis, results['reconstruction_errors'], label='Reconstruction Error', color='purple', linewidth=0.8)
        if 'threshold' in results:
            ax2.axhline(y=results['threshold'], color='red', linestyle='--', label='Threshold', linewidth=2)
        ax2.set_ylabel('MSE')
        ax2.set_title('Reconstruction Error')
    elif 'anomaly_scores' in results:
        ax2.plot(time_axis, -results['anomaly_scores'], label='Anomaly Score (Inverted)', color='green', linewidth=0.8)
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Anomaly Score (Higher = More Anomalous)')
    
    ax2.set_xlabel('Time / Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化结果已保存: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("异常检测模型测试程序")
    print("="*60)
    
    # 1. 加载模型
    model, scaler, config, threshold_info = load_model_and_config(MODEL_TYPE, MODEL_TIMESTAMP)
    
    # 2. 加载数据
    print(f"\n加载数据: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, parse_dates=['时间'] if '时间' in pd.read_csv(DATA_FILE, nrows=1).columns else [])
    print(f"✓ 数据加载成功: {df.shape}")
    
    # 3. 进行异常检测
    print(f"\n{'='*60}")
    print("开始异常检测...")
    print(f"{'='*60}")
    
    if MODEL_TYPE == 'random_forest':
        results = detect_anomalies_rf(model, df, config, threshold_info, START_IDX, END_IDX)
    elif MODEL_TYPE == 'isolation_forest':
        results = detect_anomalies_if(model, scaler, df, config, START_IDX, END_IDX)
    elif MODEL_TYPE == 'lstm':
        results = detect_anomalies_lstm(model, scaler, df, config, threshold_info, START_IDX, END_IDX)
    elif MODEL_TYPE == 'lstm_ae':
        results = detect_anomalies_lstm_ae(model, scaler, df, config, threshold_info, START_IDX, END_IDX)
    elif MODEL_TYPE == 'lstm_vae':
        results = detect_anomalies_lstm_vae(model, scaler, df, config, threshold_info, START_IDX, END_IDX)
    
    # 4. 可视化结果
    print(f"\n{'='*60}")
    print("生成可视化...")
    print(f"{'='*60}")
    
    save_path = None
    if SAVE_RESULTS:
        os.makedirs(RESULT_DIR, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RESULT_DIR, f'{MODEL_TYPE}_test_{timestamp}.png')
    
    visualize_results(results, MODEL_TYPE, save_path)
    
    # 5. 保存检测结果
    if SAVE_RESULTS:
        result_file = os.path.join(RESULT_DIR, f'{MODEL_TYPE}_anomalies_{timestamp}.csv')
        result_df = pd.DataFrame({
            'time': results['time'],
            'actual': results['actual'],
            'is_anomaly': results['is_anomaly']
        })
        
        if 'predictions' in results:
            result_df['predicted'] = results['predictions']
            result_df['residual'] = results['residuals']
        elif 'reconstructed' in results:
            result_df['reconstructed'] = results['reconstructed']
            result_df['reconstruction_error'] = results['reconstruction_errors']
        elif 'anomaly_scores' in results:
            result_df['anomaly_score'] = results['anomaly_scores']
        
        result_df.to_csv(result_file, index=False, encoding='utf-8')
        print(f"✓ 检测结果已保存: {result_file}")
    
    # 6. 打印统计信息
    print(f"\n{'='*60}")
    print("检测统计")
    print(f"{'='*60}")
    print(f"总样本数: {len(results['is_anomaly'])}")
    print(f"异常样本数: {results['is_anomaly'].sum()}")
    print(f"异常比例: {results['is_anomaly'].sum()/len(results['is_anomaly'])*100:.2f}%")
    
    # 找出异常点的索引
    anomaly_indices = np.where(results['is_anomaly'] == 1)[0]
    if len(anomaly_indices) > 0:
        print(f"\n前10个异常点的索引:")
        for idx in anomaly_indices[:10]:
            print(f"  索引 {idx}: 实际值 = {results['actual'][idx]:.4f}")
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
