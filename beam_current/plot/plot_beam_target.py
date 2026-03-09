#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制束流 target 时序图
数据来源: data/raw/beamdata.csv (需先运行 process_beam_data.py 生成)
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'beamdata.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plot')


def plot_beam_target():
    """绘制束流 target 时序图"""
    print(f"正在读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 解析时间列
    if '时间' in df.columns:
        df['时间'] = pd.to_datetime(df['时间'])
        x_data = df['时间']
        x_label = '时间'
    else:
        x_data = range(len(df))
        x_label = '样本索引'

    target = df['target']

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_data, target.values, linewidth=0.8, alpha=0.9, color='#2563eb', label='束流 (target)')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('束流强度', fontsize=12)
    ax.set_title('束流时序图', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f'beam_target_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")
    return save_path


if __name__ == '__main__':
    plot_beam_target()
