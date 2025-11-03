"""
束位数据可视化脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_beam_position_data():
    """绘制束位数据图表"""
    
    # 读取数据
    df = pd.read_csv('./data/束位数据.csv')
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/束位监测/position_analysis/{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 主要分析图 - 3x3布局
    fig = plt.figure(figsize=(20, 16))
    
    # 束位X时间序列
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['束位X'].values, linewidth=0.8, alpha=0.8, color='blue')
    ax1.set_title('束位X时间序列', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('束位X值')
    ax1.grid(True, alpha=0.3)
    
    # 束位Y时间序列
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df['束位Y'].values, linewidth=0.8, alpha=0.8, color='red')
    ax2.set_title('束位Y时间序列', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('束位Y值')
    ax2.grid(True, alpha=0.3)
    
    # 束位X分布直方图
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df['束位X'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(df['束位X'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'均值: {df["束位X"].mean():.1f}')
    ax3.set_title('束位X分布直方图', fontsize=14, fontweight='bold')
    ax3.set_xlabel('束位X值')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 束位Y分布直方图
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df['束位Y'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(df['束位Y'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'均值: {df["束位Y"].mean():.1f}')
    ax4.set_title('束位Y分布直方图', fontsize=14, fontweight='bold')
    ax4.set_xlabel('束位Y值')
    ax4.set_ylabel('频次')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 束位轨迹图（X vs Y）
    ax5 = plt.subplot(3, 3, 5)
    scatter = ax5.scatter(df['束位X'], df['束位Y'], alpha=0.6, s=1, 
                         c=range(len(df)), cmap='viridis')
    ax5.set_title('束位轨迹图 (X vs Y)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('束位X')
    ax5.set_ylabel('束位Y')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='时间点')
    
    # 束位X和Y的箱线图
    ax6 = plt.subplot(3, 3, 6)
    box_data = [df['束位X'], df['束位Y']]
    box_plot = ax6.boxplot(box_data, patch_artist=True, labels=['束位X', '束位Y'])
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax6.set_title('束位X和Y箱线图', fontsize=14, fontweight='bold')
    ax6.set_ylabel('束位值')
    ax6.grid(True, alpha=0.3)
    
    # 束位X移动平均
    ax7 = plt.subplot(3, 3, 7)
    window_sizes = [10, 50, 100]
    ax7.plot(df['束位X'].values, alpha=0.3, linewidth=0.5, label='原始数据', color='blue')
    for window in window_sizes:
        if len(df) > window:
            moving_avg = df['束位X'].rolling(window=window).mean()
            ax7.plot(moving_avg.values, linewidth=1.5, label=f'{window}点移动平均')
    ax7.set_title('束位X移动平均', fontsize=14, fontweight='bold')
    ax7.set_xlabel('时间点')
    ax7.set_ylabel('束位X值')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 束位Y移动平均
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(df['束位Y'].values, alpha=0.3, linewidth=0.5, label='原始数据', color='red')
    for window in window_sizes:
        if len(df) > window:
            moving_avg = df['束位Y'].rolling(window=window).mean()
            ax8.plot(moving_avg.values, linewidth=1.5, label=f'{window}点移动平均')
    ax8.set_title('束位Y移动平均', fontsize=14, fontweight='bold')
    ax8.set_xlabel('时间点')
    ax8.set_ylabel('束位Y值')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 统计信息文本
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
    束位数据统计信息:
    
    数据点数: {len(df):,}
    
    束位X:
    均值: {df['束位X'].mean():.1f}
    标准差: {df['束位X'].std():.1f}
    范围: {df['束位X'].min():.1f} ~ {df['束位X'].max():.1f}
    
    束位Y:
    均值: {df['束位Y'].mean():.1f}
    标准差: {df['束位Y'].std():.1f}
    范围: {df['束位Y'].min():.1f} ~ {df['束位Y'].max():.1f}
    
    相关性: {df['束位X'].corr(df['束位Y']):.4f}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('束位数据分析报告', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(result_dir, f"beam_position_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"束位分析图已保存为: {plot_path}")
    
    plt.show()
    
    # 2. 轨迹详细分析图
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 束位轨迹图（带时间颜色）
    scatter = ax1.scatter(df['束位X'], df['束位Y'], c=range(len(df)), 
                          cmap='viridis', alpha=0.6, s=2)
    ax1.set_title('束位轨迹图（时间演化）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('束位X')
    ax1.set_ylabel('束位Y')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='时间点')
    
    # 束位轨迹图（带速度颜色）
    dx = df['束位X'].diff()
    dy = df['束位Y'].diff()
    speed = np.sqrt(dx**2 + dy**2)
    
    scatter2 = ax2.scatter(df['束位X'], df['束位Y'], c=speed, 
                           cmap='plasma', alpha=0.6, s=2)
    ax2.set_title('束位轨迹图（速度着色）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('束位X')
    ax2.set_ylabel('束位Y')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='速度')
    
    # 束位X和Y的联合分布
    ax3.hist2d(df['束位X'], df['束位Y'], bins=50, cmap='Blues', alpha=0.8)
    ax3.set_title('束位X和Y联合分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('束位X')
    ax3.set_ylabel('束位Y')
    
    # 束位变化率时间序列
    ax4.plot(speed.values, linewidth=0.8, alpha=0.8, color='purple')
    ax4.set_title('束位变化率时间序列', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间点')
    ax4.set_ylabel('变化率')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存轨迹分析图
    trajectory_path = os.path.join(result_dir, f"beam_position_trajectory_{timestamp}.png")
    plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
    print(f"束位轨迹分析图已保存为: {trajectory_path}")
    
    plt.show()
    
    # 3. 异常值检测图
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 束位X异常值检测
    Q1_X = df['束位X'].quantile(0.25)
    Q3_X = df['束位X'].quantile(0.75)
    IQR_X = Q3_X - Q1_X
    lower_X = Q1_X - 1.5 * IQR_X
    upper_X = Q3_X + 1.5 * IQR_X
    outliers_X = df[(df['束位X'] < lower_X) | (df['束位X'] > upper_X)]
    
    ax1.plot(df['束位X'].values, linewidth=0.8, alpha=0.8, label='正常值', color='blue')
    if len(outliers_X) > 0:
        ax1.scatter(outliers_X.index, outliers_X['束位X'], color='red', s=20, 
                   alpha=0.8, label=f'异常值 ({len(outliers_X)}个)')
    ax1.axhline(y=lower_X, color='orange', linestyle='--', alpha=0.7, label='下界')
    ax1.axhline(y=upper_X, color='orange', linestyle='--', alpha=0.7, label='上界')
    ax1.set_title('束位X异常值检测', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('束位X值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 束位Y异常值检测
    Q1_Y = df['束位Y'].quantile(0.25)
    Q3_Y = df['束位Y'].quantile(0.75)
    IQR_Y = Q3_Y - Q1_Y
    lower_Y = Q1_Y - 1.5 * IQR_Y
    upper_Y = Q3_Y + 1.5 * IQR_Y
    outliers_Y = df[(df['束位Y'] < lower_Y) | (df['束位Y'] > upper_Y)]
    
    ax2.plot(df['束位Y'].values, linewidth=0.8, alpha=0.8, label='正常值', color='red')
    if len(outliers_Y) > 0:
        ax2.scatter(outliers_Y.index, outliers_Y['束位Y'], color='red', s=20, 
                   alpha=0.8, label=f'异常值 ({len(outliers_Y)}个)')
    ax2.axhline(y=lower_Y, color='orange', linestyle='--', alpha=0.7, label='下界')
    ax2.axhline(y=upper_Y, color='orange', linestyle='--', alpha=0.7, label='上界')
    ax2.set_title('束位Y异常值检测', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('束位Y值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存异常值检测图
    outlier_path = os.path.join(result_dir, f"beam_position_outliers_{timestamp}.png")
    plt.savefig(outlier_path, dpi=300, bbox_inches='tight')
    print(f"束位异常值检测图已保存为: {outlier_path}")
    
    plt.show()
    
    print(f"\n=== 绘图完成 ===")
    print(f"结果已保存到: {result_dir}")
    print("生成的图表:")
    print(f"  - 束位分析图: beam_position_analysis_{timestamp}.png")
    print(f"  - 束位轨迹图: beam_position_trajectory_{timestamp}.png")
    print(f"  - 异常值检测图: beam_position_outliers_{timestamp}.png")

if __name__ == "__main__":
    plot_beam_position_data()