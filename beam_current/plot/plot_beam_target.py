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


def _compute_mu_sigma_thresholds(
    df: pd.DataFrame,
    baseline_start_idx: int = 0,
    baseline_end_idx: int = 10000,
    sigma_k: float = 3.0,
):
    """用 baseline 段估计均值/方差，并返回阈值区间 [lower, upper]."""
    if 'target' not in df.columns:
        raise KeyError("数据中缺少 'target' 列，无法计算阈值。")

    n = len(df)
    s = max(0, int(baseline_start_idx))
    e = min(n - 1, int(baseline_end_idx))
    if n == 0 or s > e:
        raise ValueError(f"baseline 索引范围非法: {baseline_start_idx}~{baseline_end_idx}，数据长度={n}")

    baseline = pd.to_numeric(df.loc[s:e, 'target'], errors='coerce').dropna()
    if baseline.empty:
        raise ValueError("baseline 段 target 全是 NaN/无法转换为数值，无法计算阈值。")

    mu = float(baseline.mean())
    sigma = float(baseline.std(ddof=0))
    lower = mu - sigma_k * sigma
    upper = mu + sigma_k * sigma
    return mu, sigma, lower, upper


def plot_beam_target(start_idx: int = None, end_idx: int = None):
    """绘制束流 target 时序图

    Args:
        start_idx: 起始点索引（含），None 表示从头开始
        end_idx:   结束点索引（含），None 表示到末尾
    """
    print(f"正在读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 按索引切片
    if start_idx is not None or end_idx is not None:
        s = start_idx if start_idx is not None else 0
        e = (end_idx + 1) if end_idx is not None else len(df)
        df = df.iloc[s:e].reset_index(drop=True)

    # 横坐标统一使用样本索引（不使用时间列）
    offset = start_idx if start_idx is not None else 0
    x_data = range(offset, offset + len(df))
    x_label = '样本索引'

    target = df['target']

    # 标题中体现范围
    range_str = ''
    if start_idx is not None or end_idx is not None:
        s_str = str(start_idx) if start_idx is not None else '0'
        e_str = str(end_idx) if end_idx is not None else str(len(df) - 1)
        range_str = f'（第 {s_str}～{e_str} 点）'

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(list(x_data), target.values, linewidth=1.2, alpha=0.9, color='#2563eb', label='束流')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('target', fontsize=12)
    # ax.set_title(f'束流时序图{range_str}', fontsize=14)
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


def plot_beam_target_with_anomaly(
    start_idx: int = None,
    end_idx: int = None,
    baseline_start_idx: int = 0,
    baseline_end_idx: int = 10000,
    sigma_k: float = 3.0,
    show_threshold_lines: bool = True,
):
    """绘制束流 target 时序图，并区分正常/异常样本（基于 baseline 段的 ±kσ）。

    规则：用全量数据的 [baseline_start_idx, baseline_end_idx] 段计算 \(\mu,\sigma\)，
    任何点若 target ∈ [\(\mu-k\sigma\), \(\mu+k\sigma\)] 视为正常，否则为异常。
    """
    print(f"正在读取数据: {DATA_PATH}")
    df_all = pd.read_csv(DATA_PATH)
    mu, sigma, lower, upper = _compute_mu_sigma_thresholds(
        df_all,
        baseline_start_idx=baseline_start_idx,
        baseline_end_idx=baseline_end_idx,
        sigma_k=sigma_k,
    )

    # 按索引切片用于绘图（但阈值始终来自全量 baseline 段）
    df = df_all
    if start_idx is not None or end_idx is not None:
        s = start_idx if start_idx is not None else 0
        e = (end_idx + 1) if end_idx is not None else len(df)
        df = df.iloc[s:e].reset_index(drop=True)

    # 横坐标统一使用样本索引（不使用时间列）
    offset = start_idx if start_idx is not None else 0
    x_data = range(offset, offset + len(df))
    x_label = '样本索引'

    target = pd.to_numeric(df['target'], errors='coerce')
    is_normal = target.between(lower, upper, inclusive='both')
    is_anomaly = ~is_normal & target.notna()

    # 标题中体现范围
    range_str = ''
    if start_idx is not None or end_idx is not None:
        s_str = str(start_idx) if start_idx is not None else '0'
        e_str = str(end_idx) if end_idx is not None else str(len(df) - 1)
        range_str = f'（第 {s_str}～{e_str} 点）'

    fig, ax = plt.subplots(figsize=(14, 6))

    # 正常：蓝色折线（只连接正常点，避免异常点拉扯折线）
    normal_y = target.where(is_normal)
    ax.plot(list(x_data), normal_y.values, linewidth=1.2, alpha=0.9, color='#2563eb', label='正常样本')

    # 异常：红色散点（数量通常较少）
    if is_anomaly.any():
        ax.scatter(
            [x for x, flag in zip(list(x_data), is_anomaly.values) if flag],
            target[is_anomaly].values,
            s=10,
            color='#dc2626',
            alpha=0.9,
            label='异常样本',
            zorder=3,
        )

    if show_threshold_lines:
        ax.axhline(upper, color='#f59e0b', linestyle='--', linewidth=1.2, alpha=0.9, label=f'+{sigma_k}σ')
        ax.axhline(lower, color='#f59e0b', linestyle='--', linewidth=1.2, alpha=0.9, label=f'-{sigma_k}σ')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('target', fontsize=12)
    # ax.set_title(f'束流时序图（正常/异常）{range_str}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f'beam_target_anomaly_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    n_total = int(target.notna().sum())
    n_anom = int(is_anomaly.sum())
    print(
        f"阈值来自 baseline[{baseline_start_idx}~{baseline_end_idx}]："
        f"mu={mu:.6g}, sigma={sigma:.6g}, lower={lower:.6g}, upper={upper:.6g}；"
        f"绘图范围{range_str}，异常点 {n_anom}/{n_total}"
    )
    print(f"图片已保存: {save_path}")
    return save_path


if __name__ == '__main__':
    plot_beam_target(start_idx=0, end_idx=100000)
    plot_beam_target_with_anomaly(
        start_idx=0,
        end_idx=25000,
        baseline_start_idx=0,
        baseline_end_idx=10000,
        sigma_k=3.0,
        show_threshold_lines=True,
    )
