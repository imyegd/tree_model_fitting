#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测工具函数
提供基于时间范围的束流异常检测功能
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def detect_beam_anomalies(start_time, end_time):
    """
    对指定时间范围内的束流数据进行异常检测
    
    参数:
        start_time (str): 开始时间，格式如 '2024-01-01 00:00:00' 或 '2024-01-01'
        end_time (str): 结束时间，格式如 '2024-01-01 23:59:59' 或 '2024-01-01'
    
    返回:
        dict: 包含以下键值对
            - 'total_samples': 总样本数
            - 'anomaly_count': 异常样本数
            - 'anomaly_ratio': 异常比例 (%)
            - 'anomaly_indices': 异常点在数据中的索引列表
            - 'anomaly_times': 异常点对应的时间列表
            - 'anomaly_values': 异常点的实际值列表
            - 'threshold': 检测阈值
            - 'summary': 检测摘要文本
            - 'plot_file': 检测结果图片路径
            - 'csv_file': 异常点详细信息CSV文件路径
    
    示例:
        >>> result = detect_beam_anomalies('2024-01-01', '2024-01-02')
        >>> print(f"检测到 {result['anomaly_count']} 个异常点")
        >>> print(result['summary'])
        >>> print(f"图片保存在: {result['plot_file']}")
    """
    
    # 固定的配置
    data_file = 'data/raw/束流.csv'
    model_dir = './result/anomaly_detection_models'
    
    print(f"\n{'='*60}")
    print(f"束流异常检测工具")
    print(f"{'='*60}")
    print(f"时间范围: {start_time} 至 {end_time}")
    # print(f"模型: RandomForest (时间戳: {model_timestamp})")
    print(f"{'='*60}\n")
    
    # ========== 1. 加载模型和配置 ==========
    try:
        # 加载配置
        config_path = os.path.join(model_dir, f'random_forest_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ 配置加载成功")
        
        # 加载阈值信息
        threshold_path = os.path.join(model_dir, f'random_forest_threshold.json')
        with open(threshold_path, 'r', encoding='utf-8') as f:
            threshold_info = json.load(f)
        print(f"✓ 阈值信息加载成功: {threshold_info['threshold']:.6f}")
        
        # 加载模型
        model_path = os.path.join(model_dir, f'random_forest_model.pkl')
        model = joblib.load(model_path)
        print(f"✓ RandomForest模型加载成功")
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"模型文件未找到: {e}\n请确保模型已训练并保存到 {model_dir}")
    except Exception as e:
        raise Exception(f"模型加载失败: {e}")
    
    # ========== 2. 加载和过滤数据 ==========
    try:
        # 读取数据
        df = pd.read_csv(data_file)
        print(f"✓ 数据加载成功: {df.shape}")
        
        # 转换时间列
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
        else:
            raise ValueError("数据中未找到'时间'列")
        
        # 转换输入的时间字符串
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        # 按时间范围过滤数据
        mask = (df['时间'] >= start_dt) & (df['时间'] <= end_dt)
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"⚠ 警告: 指定时间范围内没有数据")
            return {
                'total_samples': 0,
                'anomaly_count': 0,
                'anomaly_ratio': 0.0,
                'anomaly_indices': [],
                'anomaly_times': [],
                'anomaly_values': [],
                'threshold': threshold_info['threshold'],
                'summary': f"时间范围 {start_time} 至 {end_time} 内无数据"
            }
        
        print(f"✓ 时间范围过滤完成: {len(df_filtered)} 个样本")
        print(f"  实际时间范围: {df_filtered['时间'].min()} 至 {df_filtered['时间'].max()}")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件未找到: {data_file}")
    except Exception as e:
        raise Exception(f"数据加载失败: {e}")
    
    # ========== 3. 进行异常检测 ==========
    try:
        features = config['features']
        target = config['target']
        threshold = threshold_info['threshold']
        
        # 提取特征和目标
        X_test = df_filtered[features].values
        y_test = df_filtered[target].values
        
        # 预测
        print(f"\n开始异常检测...")
        y_pred = model.predict(X_test)
        
        # 计算残差
        residuals = np.abs(y_test - y_pred)
        
        # 异常判定
        is_anomaly = (residuals > threshold).astype(int)
        anomaly_count = is_anomaly.sum()
        anomaly_ratio = (anomaly_count / len(df_filtered)) * 100
        
        print(f"✓ 检测完成")
        print(f"  总样本数: {len(df_filtered)}")
        print(f"  异常样本数: {anomaly_count}")
        print(f"  异常比例: {anomaly_ratio:.2f}%")
        
    except KeyError as e:
        raise KeyError(f"数据列缺失: {e}\n请确保数据文件包含所需的特征列")
    except Exception as e:
        raise Exception(f"异常检测失败: {e}")
    
    # ========== 4. 整理结果 ==========
    # 获取异常点信息
    anomaly_mask = is_anomaly == 1
    anomaly_indices = np.where(anomaly_mask)[0].tolist()
    anomaly_times = df_filtered['时间'].iloc[anomaly_indices].tolist()
    anomaly_values = y_test[anomaly_mask].tolist()
    
    # 生成摘要
    summary = f"""
    【异常检测结果摘要】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    时间范围: {start_time} 至 {end_time}
    检测模型: RandomForest
    检测阈值: {threshold:.6f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    总样本数: {len(df_filtered)}
    异常样本数: {anomaly_count}
    异常比例: {anomaly_ratio:.2f}%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    if anomaly_count > 0:
        summary += f"\n前5个异常点:\n"
        for i, (idx, time, value) in enumerate(zip(anomaly_indices[:5], anomaly_times[:5], anomaly_values[:5])):
            summary += f"  {i+1}. 时间: {time}, 值: {value:.4f}, 残差: {residuals[idx]:.4f}\n"
    else:
        summary += "\n未检测到异常点\n"
    
    summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # ========== 5. 生成可视化图片 ==========
    print(f"\n正在生成检测结果图片...")
    
    # 创建保存目录
    output_dir = './result/anoma_detection/test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f'random_forest_test_{timestamp}.png')
    csv_filename = os.path.join(output_dir, f'random_forest_anomalies_{timestamp}.csv')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 使用数值索引而非datetime对象，避免性能问题
    index_axis = np.arange(len(df_filtered))
    
    # 子图1: 实际值 vs 预测值，标注异常点
    ax1.plot(index_axis, y_test, label='实际值 (Actual)', alpha=0.7, linewidth=0.8, color='blue')
    ax1.plot(index_axis, y_pred, label='预测值 (Predicted)', alpha=0.6, linewidth=0.8, linestyle='--', color='green')
    
    # 标注异常点
    if anomaly_count > 0:
        anomaly_local_indices = np.where(anomaly_mask)[0]
        ax1.scatter(anomaly_local_indices, y_test[anomaly_mask], 
                   color='red', s=15, label=f'异常点 (共{anomaly_count}个)', zorder=5, alpha=0.8)
    
    ax1.set_title(f'RandomForest 异常检测结果 ({start_time} 至 {end_time})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间索引 (Time Index)', fontsize=11)
    ax1.set_ylabel('Target 值', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2: 残差曲线与阈值
    ax2.plot(index_axis, residuals, label='残差 (Residual)', color='orange', linewidth=0.8, alpha=0.8)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值 (Threshold = {threshold:.6f})')
    
    # 填充超过阈值的区域
    if anomaly_count > 0:
        ax2.fill_between(index_axis, 0, residuals, where=(residuals > threshold), 
                        color='red', alpha=0.2, label='异常区域')
    
    ax2.set_title('残差分析与异常阈值', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间索引 (Time Index)', fontsize=11)
    ax2.set_ylabel('残差绝对值', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ 检测结果图片已保存: {plot_filename}")
    plt.close()
    
    # ========== 6. 保存异常点详细信息到CSV ==========
    if anomaly_count > 0:
        anomaly_df = pd.DataFrame({
            '时间': anomaly_times,
            'Target实际值': anomaly_values,
            'Target预测值': y_pred[anomaly_mask],
            '残差': residuals[anomaly_mask],
            '局部索引': np.where(anomaly_mask)[0]
        })
        anomaly_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✓ 异常点详细信息已保存: {csv_filename}")
    
    # 构建返回结果
    result = {
        'total_samples': int(len(df_filtered)),
        'anomaly_count': int(anomaly_count),
        'anomaly_ratio': float(anomaly_ratio),
        'anomaly_indices': anomaly_indices,
        'anomaly_times': [str(t) for t in anomaly_times],  # 转换为字符串便于序列化
        'anomaly_values': [float(v) for v in anomaly_values],
        'threshold': float(threshold),
        'summary': summary,
        'plot_file': plot_filename,
        'csv_file': csv_filename if anomaly_count > 0 else None
    }
    
    
    print(f"\n{summary}")
    
    return result


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例: 按时间范围检测
    print("\n【束流异常检测示例】")
    result = detect_beam_anomalies(
        start_time='2023-01-01',
        end_time='2026-01-02'
    )
    
    print(f"\n结果已保存:")
    print(f"  - 可视化图片: {result['plot_file']}")
    if result['csv_file']:
        print(f"  - 异常点详情: {result['csv_file']}")
