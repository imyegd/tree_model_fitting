#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
束流数据分析脚本
分析束流变化趋势，绘制相关图表
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_data(csv_file_path):
    """
    加载束流数据
    
    Args:
        csv_file_path (str): CSV文件路径
    
    Returns:
        pd.DataFrame: 数据框
    """
    print(f"正在读取数据文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    return df

def analyze_beam_current(df):
    """
    分析束流数据的基本统计信息
    
    Args:
        df (pd.DataFrame): 数据框
    
    Returns:
        dict: 统计信息
    """
    # 找到束流列
    beam_col = None
    if '束流' in df.columns:
        beam_col = '束流'
    elif 'target' in df.columns:
        beam_col = 'target'
    else:
        raise ValueError("未找到束流列")
    
    beam_data = df[beam_col]
    
    stats = {
        'count': len(beam_data),
        'mean': beam_data.mean(),
        'std': beam_data.std(),
        'min': beam_data.min(),
        'max': beam_data.max(),
        'median': beam_data.median(),
        'q25': beam_data.quantile(0.25),
        'q75': beam_data.quantile(0.75),
        'skewness': beam_data.skew(),
        'kurtosis': beam_data.kurtosis()
    }
    
    print(f"\n=== 束流数据统计信息 ===")
    print(f"数据点数: {stats['count']}")
    print(f"均值: {stats['mean']:.6f}")
    print(f"标准差: {stats['std']:.6f}")
    print(f"最小值: {stats['min']:.6f}")
    print(f"最大值: {stats['max']:.6f}")
    print(f"中位数: {stats['median']:.6f}")
    print(f"25%分位数: {stats['q25']:.6f}")
    print(f"75%分位数: {stats['q75']:.6f}")
    print(f"偏度: {stats['skewness']:.6f}")
    print(f"峰度: {stats['kurtosis']:.6f}")
    
    return stats, beam_col

def plot_beam_current_analysis(df, beam_col, save_dir):
    """
    绘制束流分析图表
    
    Args:
        df (pd.DataFrame): 数据框
        beam_col (str): 束流列名
        save_dir (str): 保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    
    beam_data = df[beam_col]
    
    # 1. 时间序列图
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(beam_data.values, linewidth=0.8, alpha=0.8)
    ax1.set_title('束流时间序列', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('束流值')
    ax1.grid(True, alpha=0.3)
    
    # 2. 分布直方图
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(beam_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(beam_data.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {beam_data.mean():.4f}')
    ax2.axvline(beam_data.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {beam_data.median():.4f}')
    ax2.set_title('束流分布直方图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('束流值')
    ax2.set_ylabel('频次')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax3 = plt.subplot(3, 3, 3)
    box_plot = ax3.boxplot(beam_data, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax3.set_title('束流箱线图', fontsize=14, fontweight='bold')
    ax3.set_ylabel('束流值')
    ax3.grid(True, alpha=0.3)
    
    # 4. 移动平均
    window_sizes = [10, 50, 100]
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(beam_data.values, alpha=0.3, linewidth=0.5, label='原始数据')
    for window in window_sizes:
        if len(beam_data) > window:
            moving_avg = beam_data.rolling(window=window).mean()
            ax4.plot(moving_avg.values, linewidth=1.5, label=f'{window}点移动平均')
    ax4.set_title('束流移动平均', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间点')
    ax4.set_ylabel('束流值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 差分图（变化率）
    ax5 = plt.subplot(3, 3, 5)
    diff_data = beam_data.diff().dropna()
    ax5.plot(diff_data.values, linewidth=0.8, alpha=0.8, color='orange')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_title('束流一阶差分（变化率）', fontsize=14, fontweight='bold')
    ax5.set_xlabel('时间点')
    ax5.set_ylabel('变化量')
    ax5.grid(True, alpha=0.3)
    
    # 6. 累积和
    ax6 = plt.subplot(3, 3, 6)
    cumsum_data = beam_data.cumsum()
    ax6.plot(cumsum_data.values, linewidth=1.5, color='purple')
    ax6.set_title('束流累积和', fontsize=14, fontweight='bold')
    ax6.set_xlabel('时间点')
    ax6.set_ylabel('累积值')
    ax6.grid(True, alpha=0.3)
    
    # 7. 自相关图
    ax7 = plt.subplot(3, 3, 7)
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(beam_data, ax=ax7)
    ax7.set_title('束流自相关图', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. 功率谱密度
    ax8 = plt.subplot(3, 3, 8)
    from scipy import signal
    # 将Series安全转换为numpy数组，并移除NaN，避免SciPy在内部切片时报错
    beam_array = pd.to_numeric(beam_data, errors='coerce').to_numpy()
    beam_array = beam_array[~np.isnan(beam_array)]
    if beam_array.ndim != 1:
        beam_array = np.ravel(beam_array)
    if beam_array.size >= 16:  # 数据太少时跳过PSD
        nperseg = max(8, min(1024, beam_array.size // 4))
        freqs, psd = signal.welch(beam_array, nperseg=nperseg)
        ax8.semilogy(freqs, psd)
        ax8.set_title('束流功率谱密度', fontsize=14, fontweight='bold')
        ax8.set_xlabel('频率')
        ax8.set_ylabel('功率谱密度')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, '数据量不足，跳过PSD', ha='center', va='center')
        ax8.set_title('束流功率谱密度')
    
    # 9. 统计信息文本
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
    统计信息:
    
    数据点数: {len(beam_data):,}
    均值: {beam_data.mean():.6f}
    标准差: {beam_data.std():.6f}
    最小值: {beam_data.min():.6f}
    最大值: {beam_data.max():.6f}
    中位数: {beam_data.median():.6f}
    
    变异系数: {beam_data.std()/beam_data.mean()*100:.2f}%
    偏度: {beam_data.skew():.4f}
    峰度: {beam_data.kurtosis():.4f}
    
    25%分位数: {beam_data.quantile(0.25):.6f}
    75%分位数: {beam_data.quantile(0.75):.6f}
    IQR: {beam_data.quantile(0.75) - beam_data.quantile(0.25):.6f}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('束流数据分析报告', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"beam_current_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"分析图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def analyze_feature_correlation(df, beam_col, save_dir):
    """
    分析特征与束流的相关性
    
    Args:
        df (pd.DataFrame): 数据框
        beam_col (str): 束流列名
        save_dir (str): 保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取特征列
    feature_columns = [col for col in df.columns if col.startswith('feature')]
    
    if not feature_columns:
        print("未找到特征列，跳过相关性分析")
        return None
    
    # 计算相关性
    correlations = []
    for feature in feature_columns:
        corr = df[feature].corr(df[beam_col])
        correlations.append((feature, corr))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n=== 特征与束流相关性分析 ===")
    print("前10个最相关的特征:")
    for i, (feature, corr) in enumerate(correlations[:10]):
        print(f"{i+1:2d}. {feature}: {corr:.6f}")
    
    # 绘制相关性图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 相关性条形图
    features = [item[0] for item in correlations[:15]]
    corrs = [item[1] for item in correlations[:15]]
    
    colors = ['red' if c < 0 else 'blue' for c in corrs]
    bars = ax1.barh(range(len(features)), corrs, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features)
    ax1.set_xlabel('相关系数')
    ax1.set_title('特征与束流相关性（前15个）', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. 相关性热力图（前20个特征）
    top_features = [item[0] for item in correlations[:20]]
    corr_matrix = df[top_features + [beam_col]].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"feature_correlation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"相关性分析图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def detect_outliers(df, beam_col, save_dir):
    """
    检测束流异常值
    
    Args:
        df (pd.DataFrame): 数据框
        beam_col (str): 束流列名
        save_dir (str): 保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    beam_data = df[beam_col]
    
    # 使用IQR方法检测异常值
    Q1 = beam_data.quantile(0.25)
    Q3 = beam_data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = beam_data[(beam_data < lower_bound) | (beam_data > upper_bound)]
    
    print(f"\n=== 异常值检测 ===")
    print(f"IQR方法检测到 {len(outliers)} 个异常值")
    print(f"异常值比例: {len(outliers)/len(beam_data)*100:.2f}%")
    print(f"下界: {lower_bound:.6f}")
    print(f"上界: {upper_bound:.6f}")
    
    if len(outliers) > 0:
        print(f"异常值范围: {outliers.min():.6f} ~ {outliers.max():.6f}")
    
    # 绘制异常值检测图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 时间序列中的异常值
    ax1.plot(beam_data.values, linewidth=0.8, alpha=0.8, label='正常值')
    if len(outliers) > 0:
        outlier_indices = outliers.index
        ax1.scatter(outlier_indices, outliers.values, color='red', s=20, 
                   alpha=0.8, label=f'异常值 ({len(outliers)}个)')
    ax1.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label='下界')
    ax1.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7, label='上界')
    ax1.set_title('束流异常值检测（时间序列）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('束流值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图显示异常值
    box_plot = ax2.boxplot(beam_data, patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax2.set_title('束流箱线图（显示异常值）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('束流值')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"outlier_detection_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"异常值检测图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def main():
    """主函数"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/束流/data_analysis/{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    
    # 数据文件路径
    csv_file = "./data/束流.csv"
    
    try:
        # 加载数据
        df = load_data(csv_file)
        
        # 分析束流数据
        stats, beam_col = analyze_beam_current(df)
        
        # 绘制束流分析图
        analysis_plot = plot_beam_current_analysis(df, beam_col, result_dir)
        
        # 分析特征相关性
        correlation_plot = analyze_feature_correlation(df, beam_col, result_dir)
        
        # 检测异常值
        outlier_plot = detect_outliers(df, beam_col, result_dir)
        
        # 保存统计信息
        stats_file = os.path.join(result_dir, f"beam_statistics_{timestamp}.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("束流数据统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("基本统计信息:\n")
            f.write("-" * 30 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.6f}\n")
            
            f.write(f"\n数据文件: {csv_file}\n")
            f.write(f"分析时间: {timestamp}\n")
            f.write("=" * 50 + "\n")
        
        print(f"\n=== 分析完成 ===")
        print(f"结果已保存到: {result_dir}")
        print("生成的文件:")
        print(f"  - 束流分析图: {os.path.basename(analysis_plot)}")
        if correlation_plot:
            print(f"  - 相关性分析图: {os.path.basename(correlation_plot)}")
        print(f"  - 异常值检测图: {os.path.basename(outlier_plot)}")
        print(f"  - 统计报告: {os.path.basename(stats_file)}")
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_feature4_top100_stats(df, save_dir):
    """
    绘制 feature4 前100个点的统计图（时间序列、直方图、箱线图与统计信息）
    
    Args:
        df (pd.DataFrame): 数据框，需包含 'feature4' 列
        save_dir (str): 图片保存目录
    """
    from datetime import datetime
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if 'feature4' not in df.columns:
        raise ValueError("数据中未找到列: 'feature4'")

    # 取前100个有效数值
    s_all = pd.to_numeric(df['feature2'], errors='coerce')
    s = s_all.dropna().head(100)
    if s.empty:
        raise ValueError("feature4 前100个点均为缺失或无效值，无法绘图。")

    # os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = plt.figure(figsize=(16, 12))

    # 1. 时间序列
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(s.values, linewidth=1.0, alpha=0.9, color='tab:blue')
    ax1.set_title('feature4 前100点 - 时间序列', fontsize=13, fontweight='bold')
    ax1.set_xlabel('样本序号')
    ax1.set_ylabel('值')
    ax1.grid(True, alpha=0.3)

    # 2. 分布直方图 + KDE
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(s, bins=30, kde=True, color='skyblue', edgecolor='black', ax=ax2)
    ax2.axvline(s.mean(), color='red', linestyle='--', linewidth=1.5, label=f'均值: {s.mean():.4f}')
    ax2.axvline(s.median(), color='green', linestyle='--', linewidth=1.5, label=f'中位数: {s.median():.4f}')
    ax2.set_title('feature4 前100点 - 分布直方图', fontsize=13, fontweight='bold')
    ax2.set_xlabel('值')
    ax2.set_ylabel('频次')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 箱线图（显示异常值）
    ax3 = plt.subplot(2, 2, 3)
    box_plot = ax3.boxplot(s.values, patch_artist=True, showfliers=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax3.set_title('feature4 前100点 - 箱线图', fontsize=13, fontweight='bold')
    ax3.set_ylabel('值')
    ax3.grid(True, alpha=0.3)

    # 4. 统计信息（文本）
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = f"""
    feature4 前100点统计:
    
    样本数: {len(s)}
    均值: {s.mean():.6f}
    标准差: {s.std():.6f}
    最小值: {s.min():.6f}
    最大值: {s.max():.6f}
    中位数: {s.median():.6f}
    25%分位数: {s.quantile(0.25):.6f}
    75%分位数: {s.quantile(0.75):.6f}
    IQR: {(s.quantile(0.75) - s.quantile(0.25)):.6f}
    偏度: {s.skew():.6f}
    峰度: {s.kurtosis():.6f}
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.85))

    plt.suptitle('feature4 前100点统计图', fontsize=18, fontweight='bold')
    plt.tight_layout()

    # out_path = os.path.join(save_dir, f"feature4_top100_analysis_{timestamp}.png")
    # plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # print(f"feature4 前100点统计图已保存为: {out_path}")
    plt.show()
    # return out_path


if __name__ == "__main__":
    """主函数"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/束流/data_analysis/{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    
    # 数据文件路径
    csv_file = "./data/束流.csv"
    try:
        # 加载数据
        df = load_data(csv_file)
        plot_feature4_top100_stats(df, result_dir)
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    # # main()
    #     """主函数"""
    # # 创建结果目录
    # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # result_dir = f"./result/束流/data_analysis/{timestamp}"
    # # if not os.path.exists(result_dir):
    # #     os.makedirs(result_dir)
    # #     print(f"创建结果目录: {result_dir}")
    
    # # 数据文件路径
    # csv_file = "./data/束流.csv"
    # try:
    #     # 加载数据
    #     df = load_data(csv_file)
    #     plot_feature4_top100_stats(df, result_dir)
    # except Exception as e:
    #     print(f"程序执行过程中出现错误: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

