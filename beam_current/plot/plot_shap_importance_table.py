"""
从论文表中的 Top10 指标绘制竖向柱形图（窄版，便于 LaTeX 单栏/插图排版）。
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 纵轴标签中文 + 宋体（Windows 常见名称 SimSun）
matplotlib.rcParams["font.sans-serif"] = ["SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 表 \ref{tab:shap_importance}（降序：最重要在左）
FEATURES_AND_SHAP = [
    ("feature4", 0.174422),
    ("feature10", 0.015622),
    ("feature21", 0.004532),
    ("feature2", 0.002697),
    ("feature13", 0.002058),
    ("feature19", 0.002056),
    ("feature7", 0.000658),
    ("feature6", 0.000504),
    ("feature3", 0.000493),
    ("feature11", 0.000102),
]

# 表 \ref{tab:ae_reconstruction}（升序排名；最重要在左）
FEATURES_AND_RECON_ERR = [
    ("feature4", 214.686174),
    ("feature2", 23.035921),
    ("feature15", 8.480762),
    ("feature10", 6.845333),
    ("feature7", 6.178373),
    ("feature6", 4.170542),
    ("feature28", 3.797481),
    ("feature32", 3.736328),
    ("feature31", 3.413365),
    ("feature3", 3.064298),
]

# 表 \ref{tab:pls_contribution}（降序：最重要在左）
FEATURES_AND_PLS_WEIGHT = [
    ("feature19", 0.163135),
    ("feature7", 0.154971),
    ("feature4", 0.124549),
    ("feature11", 0.103818),
    ("feature10", 0.092762),
    ("feature13", 0.062605),
    ("feature21", 0.056413),
    ("feature5", 0.053697),
    ("feature20", 0.024172),
    ("feature6", 0.018928),
]

# 表 \ref{tab:statistical_difference}（按 |z| 降序：最重要在左）
FEATURES_AND_Z_ABS = [
    ("feature4", 16.527240),
    ("feature21", 3.620657),
    ("feature2", 3.417645),
    ("feature10", 2.581961),
    ("feature11", 1.805008),
    ("feature17", 1.583015),
    ("feature19", 1.364543),
    ("feature6", 0.884279),
    ("feature7", 0.883742),
    ("feature5", 0.777267),
]

SHAP_BLUE = "#008bfb"

# 英寸；约 10.8 cm 宽，适合单栏内嵌图
FIG_WIDTH_IN = 4.25
FIG_HEIGHT_IN = 3.85


def plot_vertical(
    out_path: Path,
    *,
    pairs,
    ylabel: str,
) -> None:
    names = [f for f, _ in pairs]
    values = [float(v) for _, v in pairs]
    x = np.arange(len(names), dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), layout="constrained")
    ax.bar(x, values, color=SHAP_BLUE, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9, fontfamily="sans-serif")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.05)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "result" / "anoma_detection"
    base.mkdir(parents=True, exist_ok=True)

    shap_path = base / "rf_shap_diagnosis_table_top10_vertical.png"
    plot_vertical(
        shap_path,
        pairs=FEATURES_AND_SHAP,
        ylabel="SHAP平均绝对值",
    )
    print(f"已保存: {shap_path}")

    ae_path = base / "ae_reconstruction_error_top10_vertical.png"
    plot_vertical(
        ae_path,
        pairs=FEATURES_AND_RECON_ERR,
        ylabel="平均重构误差",
    )
    print(f"已保存: {ae_path}")

    pls_path = base / "pls_contribution_weight_top10_vertical.png"
    plot_vertical(
        pls_path,
        pairs=FEATURES_AND_PLS_WEIGHT,
        ylabel="特征权重",
    )
    print(f"已保存: {pls_path}")

    z_path = base / "statistical_difference_z_top10_vertical.png"
    plot_vertical(
        z_path,
        pairs=FEATURES_AND_Z_ABS,
        ylabel="|z|绝对值",
    )
    print(f"已保存: {z_path}")


if __name__ == "__main__":
    main()
