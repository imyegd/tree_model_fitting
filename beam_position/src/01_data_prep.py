# src/01_data_prep.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
# 导入我们刚刚创建的工具函数
import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.common.utils import reconstruct_datetime
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义文件路径
RAW_DATA_PATH = './data/raw'
PROCESSED_DATA_PATH = './data/processed'
OUTLIER_RESULT_PATH = './result/outlier_analysis'
MONITOR_FILE = os.path.join(RAW_DATA_PATH, "束位监测数据.csv")
POSITION_FILE = os.path.join(RAW_DATA_PATH, "束位数据.csv")

# 确保输出目录存在
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(OUTLIER_RESULT_PATH, exist_ok=True)

def load_and_preprocess_raw_data():
    """加载并进行初始时间戳处理和坐标差计算。"""
    print("--- 1. 数据加载与时间预处理 ---")
    try:
        beam_monitor_df = pd.read_csv(MONITOR_FILE)
        beam_position_df = pd.read_csv(POSITION_FILE)
    except FileNotFoundError as e:
        print(f"错误：找不到文件。请确保原始文件位于 {RAW_DATA_PATH} 目录下。")
        raise e

    # 监测数据时间处理
    beam_monitor_df['时间'] = pd.to_datetime(beam_monitor_df['时间'])
    beam_monitor_df = beam_monitor_df.sort_values(by='时间').reset_index(drop=True)
    
    # 束位数据时间处理和日期推断
    beam_position_df['时间_only'] = pd.to_datetime(
        beam_position_df['时间'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    
    start_date_monitor = beam_monitor_df['时间'].min().date()
    full_datetimes = reconstruct_datetime(beam_position_df['时间_only'], start_date_monitor)
    beam_position_df['完整时间'] = full_datetimes
    beam_position_df = beam_position_df.sort_values(by='完整时间').reset_index(drop=True)

    # 计算目标变量 (Delta X, Delta Y)
    beam_position_df['Delta_X'] = beam_position_df['束位X'].diff()
    beam_position_df['Delta_Y'] = beam_position_df['束位Y'].diff()
    # beam_position_df.dropna(subset=['Delta_X', 'Delta_Y'], inplace=True)
    # beam_position_df.reset_index(drop=True, inplace=True)

    # 创建时间窗口查找表 [开始时间, 结束时间)
    target_lookup_df = beam_position_df[['完整时间', 'Delta_X', 'Delta_Y']].copy()
    target_lookup_df['开始时间'] = target_lookup_df['完整时间'].shift(1)
    target_lookup_df.dropna(subset=['开始时间'], inplace=True)
    target_lookup_df.rename(columns={'完整时间': '结束时间'}, inplace=True)
    target_lookup_df.reset_index(drop=True, inplace=True)
    
    return beam_monitor_df, target_lookup_df

# -----------------------------------------------------------
# 特征工程函数
# -----------------------------------------------------------
def extract_raw_features(monitor_df, target_lookup_df, max_length=100):
    """
    提取原始特征序列 (时间窗口内的原始数据，填充到固定长度)
    
    参数:
        monitor_df: 监测数据
        target_lookup_df: 目标变量查找表
        max_length: 序列的最大长度（固定长度）
    
    返回:
        包含原始特征序列的DataFrame
    """
    print("--- 2a. 特征工程: 原始序列提取 ---")
    
    # 确定要处理的特征列
    feature_columns = [col for col in monitor_df.columns if col.startswith('feature')]
    
    final_data_list = []
    
    # 遍历目标窗口
    for index, row in target_lookup_df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        
        # 筛选当前时间窗口内的监测数据
        interval_data = monitor_df[
            (monitor_df['时间'] >= start_time) & (monitor_df['时间'] < end_time)
        ]
        
        data_row = {'Delta_X': row['Delta_X'], 'Delta_Y': row['Delta_Y']}
        
        if len(interval_data) > 0:
            # 取前 max_length 个时间点（或全部如果不足）
            interval_data = interval_data.head(max_length)
            
            # 对每个特征，将时间序列展平
            for feature_name in feature_columns:
                feature_values = interval_data[feature_name].values
                
                # 填充或截断到固定长度
                if len(feature_values) < max_length:
                    # 补零填充
                    padded_values = np.pad(feature_values, (0, max_length - len(feature_values)), 
                                          mode='constant', constant_values=0)
                else:
                    padded_values = feature_values[:max_length]
                
                # 为每个时间步创建一个特征列
                for t in range(max_length):
                    data_row[f"{feature_name}_t{t}"] = padded_values[t]
        else:
            # 如果窗口内没有监测数据，全部填充为0
            for feature_name in feature_columns:
                for t in range(max_length):
                    data_row[f"{feature_name}_t{t}"] = 0.0
        
        final_data_list.append(data_row)
    
    final_df = pd.DataFrame(final_data_list)
    print(f"  -> 原始特征维度: {len(feature_columns)} 个特征 × {max_length} 时间步 = {len(feature_columns) * max_length} 维")
    print(f"  -> 最终数据集共 {len(final_df)} 个样本。")
    return final_df


def extract_static_features(monitor_df, target_lookup_df):
    """
    提取统计特征 (均值、方差等)
    
    参数:
        monitor_df: 监测数据
        target_lookup_df: 目标变量查找表
    
    返回:
        包含统计特征的DataFrame
    """
    print("--- 2b. 特征工程: 统计量计算 ---")
    
    # 确定要处理的特征列
    feature_columns = [col for col in monitor_df.columns if col.startswith('feature')]
    STAT_FUNCTIONS = ['mean', 'std', 'min', 'max', 'median']
    
    final_data_list = []
    
    # 遍历目标窗口，计算统计特征
    for index, row in target_lookup_df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        
        # 筛选当前时间窗口内的监测数据
        interval_data = monitor_df[
            (monitor_df['时间'] >= start_time) & (monitor_df['时间'] < end_time)
        ]
        
        data_row = {'Delta_X': row['Delta_X'], 'Delta_Y': row['Delta_Y']}
        
        if len(interval_data) > 0:
            # 计算所有特征列的统计量
            stat_features = interval_data[feature_columns].agg(STAT_FUNCTIONS).transpose()
            
            # 展平并命名新的特征
            for feature_name in feature_columns:
                for stat_func in STAT_FUNCTIONS:
                    new_col_name = f"{feature_name}_{stat_func}"
                    data_row[new_col_name] = stat_features.loc[feature_name, stat_func]
        else:
            # 如果窗口内没有监测数据，则用 0 填充所有统计特征
            for feature_name in feature_columns:
                for stat_func in STAT_FUNCTIONS:
                    new_col_name = f"{feature_name}_{stat_func}"
                    data_row[new_col_name] = 0.0 
        
        final_data_list.append(data_row)

    final_df = pd.DataFrame(final_data_list)
    print(f"  -> 统计特征维度: {len(feature_columns) * len(STAT_FUNCTIONS)} 维 (44 × 5 = 220)")
    print(f"  -> 最终数据集共 {len(final_df)} 个样本。")
    return final_df


def detect_and_remove_outliers(final_df, multiplier=1.5, visualize=True):
    """
    使用IQR方法检测并移除离群点
    
    参数:
        final_df: 包含特征和目标变量的完整数据集
        multiplier: IQR倍数，默认1.5（标准）
        visualize: 是否生成可视化图表
    
    返回:
        清洗后的数据集
    """
    print("--- 2.5. 离群点检测与移除 ---")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 分离特征和目标
    y_data = final_df[['Delta_X', 'Delta_Y']].copy()
    
    # 对Delta_X检测离群点
    Q1_x = y_data['Delta_X'].quantile(0.25)
    Q3_x = y_data['Delta_X'].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    lower_x = Q1_x - multiplier * IQR_x
    upper_x = Q3_x + multiplier * IQR_x
    outliers_x = (y_data['Delta_X'] < lower_x) | (y_data['Delta_X'] > upper_x)
    
    print(f"\nDelta_X 离群点检测 (IQR方法, multiplier={multiplier}):")
    print(f"  Q1: {Q1_x:.3f}, Q3: {Q3_x:.3f}, IQR: {IQR_x:.3f}")
    print(f"  下界: {lower_x:.3f}, 上界: {upper_x:.3f}")
    print(f"  离群点数量: {outliers_x.sum()}")
    
    # 对Delta_Y检测离群点
    Q1_y = y_data['Delta_Y'].quantile(0.25)
    Q3_y = y_data['Delta_Y'].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    lower_y = Q1_y - multiplier * IQR_y
    upper_y = Q3_y + multiplier * IQR_y
    outliers_y = (y_data['Delta_Y'] < lower_y) | (y_data['Delta_Y'] > upper_y)
    
    print(f"\nDelta_Y 离群点检测 (IQR方法, multiplier={multiplier}):")
    print(f"  Q1: {Q1_y:.3f}, Q3: {Q3_y:.3f}, IQR: {IQR_y:.3f}")
    print(f"  下界: {lower_y:.3f}, 上界: {upper_y:.3f}")
    print(f"  离群点数量: {outliers_y.sum()}")
    
    # 合并离群点
    outliers_combined = outliers_x | outliers_y
    print(f"\n合并后总离群点数: {outliers_combined.sum()} ({outliers_combined.sum()/len(final_df)*100:.2f}%)")
    
    # 显示离群点详情
    if outliers_combined.sum() > 0:
        print("\n离群点详情:")
        outlier_details = y_data[outliers_combined].copy()
        outlier_details['index'] = outlier_details.index
        print(outlier_details[['index', 'Delta_X', 'Delta_Y']].to_string())
        
        # 保存离群点详情
        outlier_file = os.path.join(OUTLIER_RESULT_PATH, f'outlier_details_{timestamp}.csv')
        outlier_details[['index', 'Delta_X', 'Delta_Y']].to_csv(outlier_file, index=False)
        print(f"\n离群点详情已保存: {outlier_file}")
    
    # 可视化
    if visualize and outliers_combined.sum() > 0:
        _visualize_outliers(y_data, outliers_x, outliers_y, outliers_combined,
                           (lower_x, upper_x), (lower_y, upper_y), timestamp)
    
    # 移除离群点
    cleaned_df = final_df[~outliers_combined].reset_index(drop=True)
    
    print(f"\n数据清洗结果:")
    print(f"  原始样本数: {len(final_df)}")
    print(f"  清洗后样本数: {len(cleaned_df)}")
    print(f"  移除样本数: {outliers_combined.sum()}")
    
    return cleaned_df


def _visualize_outliers(y_data, outliers_x, outliers_y, outliers_combined, 
                       bounds_x, bounds_y, timestamp):
    """离群点可视化（内部函数）"""
    print("  -> 生成离群点可视化...")
    
    # 创建二维散点图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    normal = ~outliers_combined
    ax.scatter(y_data[normal]['Delta_X'], y_data[normal]['Delta_Y'], 
               c='blue', alpha=0.6, s=50, label=f'正常点 ({normal.sum()})')
    ax.scatter(y_data[outliers_combined]['Delta_X'], y_data[outliers_combined]['Delta_Y'], 
               c='red', alpha=0.8, s=100, marker='x', linewidths=2, 
               label=f'离群点 ({outliers_combined.sum()})')
    
    # 添加边界线
    ax.axvline(x=bounds_x[0], color='r', linestyle='--', alpha=0.5, label='Delta_X 边界')
    ax.axvline(x=bounds_x[1], color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=bounds_y[0], color='g', linestyle='--', alpha=0.5, label='Delta_Y 边界')
    ax.axhline(y=bounds_y[1], color='g', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Delta_X', fontsize=14)
    ax.set_ylabel('Delta_Y', fontsize=14)
    ax.set_title('离群点检测结果 (IQR方法)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(OUTLIER_RESULT_PATH, f'outlier_detection_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  -> 可视化图表已保存: {output_file}")
    plt.close()

    
def split_and_save_data(final_df, feature_type='static', split_type='random', test_size=0.2):
    """
    划分数据集并保存为 CSV 文件
    
    参数:
        final_df: 包含特征和目标的完整数据集
        feature_type: 特征类型 ('raw' 或 'static')
        split_type: 划分方式 ('time' 或 'random')
        test_size: 测试集比例
    """
    X = final_df.drop(columns=['Delta_X', 'Delta_Y'])
    y = final_df[['Delta_X', 'Delta_Y']]
    
    # 根据划分方式选择不同的划分策略
    if split_type == 'time':
        # 按时间顺序划分（前80%训练，后20%测试）
        split_point = int(len(final_df) * (1 - test_size))
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        split_name = 'timesplit'
    else:  # random
        # 随机划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        split_name = 'randomsplit'
    
    # 构建文件名
    X_train_file = os.path.join(PROCESSED_DATA_PATH, f'X_train_{feature_type}_{split_name}.csv')
    X_test_file = os.path.join(PROCESSED_DATA_PATH, f'X_test_{feature_type}_{split_name}.csv')
    y_train_file = os.path.join(PROCESSED_DATA_PATH, f'y_train_{feature_type}_{split_name}.csv')
    y_test_file = os.path.join(PROCESSED_DATA_PATH, f'y_test_{feature_type}_{split_name}.csv')
    
    # 保存文件
    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)
    
    print(f"  -> [{feature_type}_{split_name}] 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def process_and_save_all_combinations(raw_df, static_df, test_size=0.2):
    """
    处理并保存所有特征类型和划分方式的组合
    
    参数:
        raw_df: 原始特征数据集
        static_df: 统计特征数据集
        test_size: 测试集比例
    """
    print("\n" + "="*70)
    print("3. 数据划分和保存 - 生成所有组合")
    print("="*70)
    
    datasets = {
        'raw': raw_df,
        'static': static_df
    }
    
    split_types = ['random', 'time']
    
    saved_files = []
    
    for feature_type, df in datasets.items():
        print(f"\n处理 {feature_type.upper()} 特征:")
        for split_type in split_types:
            X_train, X_test, y_train, y_test = split_and_save_data(
                df, feature_type=feature_type, split_type=split_type, test_size=test_size
            )
            saved_files.extend([
                f'X_train_{feature_type}_{split_type}split.csv',
                f'X_test_{feature_type}_{split_type}split.csv',
                f'y_train_{feature_type}_{split_type}split.csv',
                f'y_test_{feature_type}_{split_type}split.csv'
            ])
    
    print("\n" + "="*70)
    print(f"✓ 共生成 {len(saved_files)} 个数据文件:")
    print("="*70)
    
    # 分组显示
    print("\n原始特征 (raw):")
    for f in sorted([f for f in saved_files if 'raw' in f]):
        print(f"  - {f}")
    
    print("\n统计特征 (static):")
    for f in sorted([f for f in saved_files if 'static' in f]):
        print(f"  - {f}")
    
    return saved_files
    
if __name__ == '__main__':
    # 主流程
    print("\n" + "="*80)
    print(" " * 25 + "数据预处理完整流程")
    print("="*80 + "\n")
    
    # 1. 加载和预处理原始数据
    monitor_df, target_lookup_df = load_and_preprocess_raw_data()
    
    # 2a. 提取原始特征
    raw_df = extract_raw_features(monitor_df, target_lookup_df, max_length=100)
    
    # 2b. 提取统计特征
    static_df = extract_static_features(monitor_df, target_lookup_df)
    
    # 2.5a. 对原始特征进行离群点检测与移除
    print("\n" + "-"*70)
    print("对 RAW 特征进行离群点检测")
    print("-"*70)
    raw_cleaned_df = detect_and_remove_outliers(raw_df, multiplier=1.5, visualize=False)
    
    # 2.5b. 对统计特征进行离群点检测与移除
    print("\n" + "-"*70)
    print("对 STATIC 特征进行离群点检测")
    print("-"*70)
    static_cleaned_df = detect_and_remove_outliers(static_df, multiplier=1.5, visualize=True)
    
    # 3. 生成所有特征类型和划分方式的组合
    saved_files = process_and_save_all_combinations(
        raw_cleaned_df, 
        static_cleaned_df, 
        test_size=0.2
    )
    
    print("\n" + "="*80)
    print(" " * 25 + "数据预处理流程完成！")
    print("="*80)
    print(f"\n所有文件已保存至: {PROCESSED_DATA_PATH}")
    print(f"离群点分析结果保存至: {OUTLIER_RESULT_PATH}\n")