#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已有异常诊断结果，生成 topK 特征重合率汇总表（CSV + PNG）。

输入（默认）:
  - result/anomaly_diagnose/shap_rf_anomaly.csv              (feature, mean_abs_shap)
  - result/anomaly_diagnose/pls_feature_contribution.csv     (feature, pls_weight)
  - result/anomaly_diagnose/stat_diff_results.csv            (feature, abs_z)
  - result/anomaly_diagnose/ae_feature_reconstruction_error_torch.csv (feature, mean_reconstruction_error)

输出（默认写入 result/anomaly_diagnose/）:
  - overlap_summary.csv
  - overlap_summary.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# 中文字体设置（与 plot_beam_target.py 保持一致）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "anomaly_diagnose")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    filename: str
    feature_col: str
    score_col: str
    ascending: bool = False  # False 表示分数越大越重要


METHOD_SPECS: Sequence[MethodSpec] = (
    MethodSpec(
        name="SHAP(RF)",
        filename="shap_rf_anomaly.csv",
        feature_col="feature",
        score_col="mean_abs_shap",
        ascending=False,
    ),
    MethodSpec(
        name="PLS",
        filename="pls_feature_contribution.csv",
        feature_col="feature",
        score_col="pls_weight",
        ascending=False,
    ),
    MethodSpec(
        name="StatDiff",
        filename="stat_diff_results.csv",
        feature_col="feature",
        score_col="abs_z",
        ascending=False,
    ),
    MethodSpec(
        name="AE",
        filename="ae_feature_reconstruction_error_torch.csv",
        feature_col="feature",
        score_col="mean_reconstruction_error",
        ascending=False,
    ),
)


def _load_top_features(spec: MethodSpec, top_k: int) -> List[str]:
    path = os.path.join(RESULT_DIR, spec.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到结果文件: {path}")

    df = pd.read_csv(path)
    if spec.feature_col not in df.columns:
        raise KeyError(f"{spec.filename} 缺少列 {spec.feature_col!r}，实际列: {list(df.columns)}")
    if spec.score_col not in df.columns:
        raise KeyError(f"{spec.filename} 缺少列 {spec.score_col!r}，实际列: {list(df.columns)}")

    df = df[[spec.feature_col, spec.score_col]].copy()
    df[spec.feature_col] = df[spec.feature_col].astype(str)
    df[spec.score_col] = pd.to_numeric(df[spec.score_col], errors="coerce")
    df = df.dropna(subset=[spec.score_col])
    df = df.sort_values(spec.score_col, ascending=spec.ascending)
    return df[spec.feature_col].head(top_k).tolist()


def _intersection_features(all_top_lists: Sequence[Sequence[str]], k: int) -> List[str]:
    """四种方法同时取 topK 后的交集特征。"""
    if k <= 0:
        return []
    inter = set(all_top_lists[0][:k])
    for tops in all_top_lists[1:]:
        inter &= set(tops[:k])
    return sorted(inter)


def _pairwise_overlap(a: Sequence[str], b: Sequence[str], k: int) -> Tuple[float, List[str]]:
    """两种方法 topK 的交集及重合率（|A∩B| / K）。"""
    if k <= 0:
        return 0.0, []
    sa, sb = set(a[:k]), set(b[:k])
    inter = sorted(sa & sb)
    rate = len(inter) / float(k)
    return rate, inter


def build_overlap_summary(top_ks: Sequence[int] = (3, 5, 10)) -> pd.DataFrame:
    # 预先取每个方法的 top(maxK)
    max_k = max(top_ks)
    tops: Dict[str, List[str]] = {spec.name: _load_top_features(spec, max_k) for spec in METHOD_SPECS}
    methods = [spec.name for spec in METHOD_SPECS]

    all_top_lists = [tops[m] for m in methods]

    rows: List[Dict[str, object]] = []

    # 四方法共同重合（交集），按 top3/5/10 各一行
    for k in top_ks:
        feats = _intersection_features(all_top_lists, k)
        rows.append(
            {
                "topK": k,
                "methods": " ∩ ".join(methods),
                "pair_type": "all4",
                "overlap_count": len(feats),
                "overlap_rate": (len(feats) / float(k)) if k > 0 else 0.0,
                "overlap_features": ", ".join(feats) if feats else "",
            }
        )

    # 两两 top10 重合（只算 top10）
    k_pair = 10
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            rate, feats = _pairwise_overlap(tops[m1], tops[m2], k_pair)
            rows.append(
                {
                    "topK": k_pair,
                    "methods": f"{m1} ∩ {m2}",
                    "pair_type": "pair",
                    "overlap_count": len(feats),
                    "overlap_rate": rate,
                    "overlap_features": ", ".join(feats) if feats else "",
                }
            )

    return pd.DataFrame(rows)


def save_table_png(df: pd.DataFrame, out_png: str, title: str = "四种异常诊断方法 TopK 共同重合汇总"):
    # 为了让 PNG 表格可读，限制单元格字符串长度
    df_show = df.copy()
    for c in df_show.columns:
        if c == "overlap_rate":
            df_show[c] = df_show[c].map(lambda x: f"{float(x):.0%}")
        elif c == "overlap_features":
            df_show[c] = df_show[c].map(lambda s: (s[:80] + "…") if isinstance(s, str) and len(s) > 80 else s)

    # 估计画布尺寸：列多就加宽，行多就加高
    nrows, ncols = df_show.shape
    fig_w = max(12, 1.6 * ncols)
    fig_h = max(3, 0.55 * (nrows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    top_ks = (3, 5, 10)
    df = build_overlap_summary(top_ks=top_ks)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, "overlap_summary.csv")
    out_png = os.path.join(RESULT_DIR, "overlap_summary.png")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_table_png(df, out_png)

    print(f"已生成: {out_csv}")
    print(f"已生成: {out_png}")


if __name__ == "__main__":
    main()

