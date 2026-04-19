#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载已保存的 MLP 模型，无需重新训练，直接生成预测分析图。

运行方式（从 beam_current/ 目录执行）：
    python plot/regression/mlp.py
    python plot/regression/mlp.py --model result/mlp/20260309_103227/mlp_model_20260309_103236.pkl
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════
#  配置（直接修改这里，或通过命令行参数覆盖）
# ══════════════════════════════════════════════════════════
CSV_FILE  = "data/raw/beamdata.csv"
TEST_SIZE = 0.2   # 须与训练时一致（时间顺序切分）
# ══════════════════════════════════════════════════════════


def find_latest_model(result_root: str = "result/mlp") -> str:
    """在 result/mlp/ 下按目录名排序，找到最新的 mlp_model_*.pkl"""
    pattern = os.path.join(result_root, "*", "mlp_model_*.pkl")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"在 {result_root} 下未找到任何 mlp_model_*.pkl，请先运行 src/mlp.py 训练模型。"
        )
    return candidates[-1]



def load_data(csv_path: str):
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    if "target" in df.columns:
        target_col = "target"
    elif "束流" in df.columns:
        target_col = "束流"
    else:
        raise ValueError("未找到目标列（target / 束流）")
    X = df[feature_cols].values
    y = df[target_col].values
    print(f"  样本数: {len(y)}  特征数: {len(feature_cols)}")
    return X, y, feature_cols


def split_time(X, y, test_size: float = 0.2):
    n = len(y)
    idx = int(n * (1 - test_size))
    return X[:idx], X[idx:], y[:idx], y[idx:]


def plot_results(y_train_true, y_train_pred, y_test_true, y_test_pred,
                 save_dir: str):
    """
    三图布局（无标题）：
      (a) 顶部全宽 — 测试集时序：True Values（蓝）vs Predicted Values（红虚线）
      (b) 左下     — 散点图 + Perfect Fit + R²
      (c) 右下     — 残差直方图 + KDE + STD
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1],
                   hspace=0.42, wspace=0.32)

    x_idx = np.arange(len(y_test_true))

    # ── (a) 时序 ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(x_idx, y_test_true,
              color='#2563eb', linewidth=1.2, alpha=0.85, label='True Values')
    ax_a.plot(x_idx, y_test_pred,
              color='#dc2626', linewidth=1.2, linestyle='--', alpha=0.85,
              label='Predicted Values')
    ax_a.set_xlabel('Sample Number', fontsize=11)
    ax_a.set_ylabel('束流值', fontsize=11)
    ax_a.legend(loc='upper right', fontsize=10)
    ax_a.grid(True, alpha=0.25)
    ax_a.annotate('(a)', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=11)

    # ── (b) 散点图 ────────────────────────────────────────
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

    # ── (c) 残差直方图 + KDE ──────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    residuals = y_test_true - y_test_pred
    std_val   = residuals.std()
    bins_r    = np.histogram_bin_edges(residuals, bins=35)
    ax_c.hist(residuals, bins=bins_r, density=True,
              color='#93c5fd', edgecolor='white', linewidth=0.4, alpha=0.85)
    kde   = gaussian_kde(residuals, bw_method='scott')
    x_kde = np.linspace(residuals.min(), residuals.max(), 400)
    ax_c.plot(x_kde, kde(x_kde), color='#dc2626', linewidth=2.0)
    ax_c.text(0.97, 0.93, f'STD = {std_val:.4f}',
              transform=ax_c.transAxes, fontsize=11, ha='right',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    ax_c.set_xlabel('Residue', fontsize=11)
    ax_c.set_ylabel('Frequency', fontsize=11)
    ax_c.grid(True, alpha=0.25)
    ax_c.annotate('(c)', xy=(0.01, 0.94), xycoords='axes fraction', fontsize=11)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"mlp_analysis_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {out_path}")
    plt.show()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="加载 MLP 模型并绘图（无需重新训练）")
    parser.add_argument("--model", type=str, default=None,
                        help="指定模型 .pkl 路径；不指定则自动使用最新模型")
    parser.add_argument("--data", type=str, default=CSV_FILE,
                        help=f"CSV 数据路径（默认 {CSV_FILE}）")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help=f"测试集比例（默认 {TEST_SIZE}，须与训练时一致）")
    args = parser.parse_args()

    # 1. 定位模型
    model_path = args.model or find_latest_model()
    print(f"使用模型: {model_path}")

    model = joblib.load(model_path)

    save_dir = os.path.dirname(model_path)

    # 2. 加载数据，按时间顺序切分（确定性）
    X, y, _ = load_data(args.data)
    X_train, X_test, y_train, y_test = split_time(X, y, test_size=args.test_size)

    # 3. 用已有模型预测（不重新训练）
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    print(f"训练集: {len(y_train)} 样本  测试集: {len(y_test)} 样本")
    print(f"测试集 R²: {r2_score(y_test, y_test_pred):.4f}")

    # 5. 绘图
    plot_results(y_train, y_train_pred, y_test, y_test_pred, save_dir)


if __name__ == "__main__":
    main()
