# src/01_data_prep.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta
# 导入我们刚刚创建的工具函数
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common.utils import reconstruct_datetime
from sklearn.model_selection import train_test_split
# 定义文件路径
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
MONITOR_FILE = os.path.join(RAW_DATA_PATH, "束位监测数据.csv")
POSITION_FILE = os.path.join(RAW_DATA_PATH, "束位数据.csv")

# 确保输出目录存在
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

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
# 修改后的 feature_engineering_and_padding 函数
# -----------------------------------------------------------
def feature_engineering_and_padding(monitor_df, target_lookup_df):
    """
    进行特征收集，计算窗口内的统计特征 (均值、方差等)，并构建最终数据集。
    
    注意：此版本不再使用展平和零填充。
    """
    print("--- 2. 特征工程: 统计量计算 ---")
    
    # 确定要处理的特征列
    feature_columns = [col for col in monitor_df.columns if col.startswith('feature')]
    STAT_FUNCTIONS = ['mean', 'std', 'min', 'max', 'median']
    
    final_data_list = []
    
    # 第一次循环：遍历目标窗口，计算统计特征
    for index, row in target_lookup_df.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        
        # 1. 筛选当前时间窗口内的监测数据
        interval_data = monitor_df[
            (monitor_df['时间'] >= start_time) & (monitor_df['时间'] < end_time)
        ]
        
        data_row = {'Delta_X': row['Delta_X'], 'Delta_Y': row['Delta_Y']}
        
        if len(interval_data) > 0:
            # 2. 计算所有特征列的统计量
            stat_features = interval_data[feature_columns].agg(STAT_FUNCTIONS).transpose()
            
            # 3. 展平并命名新的特征
            for feature_name in feature_columns:
                for stat_func in STAT_FUNCTIONS:
                    new_col_name = f"{feature_name}_{stat_func}"
                    data_row[new_col_name] = stat_features.loc[feature_name, stat_func]
        else:
            # 4. 如果窗口内没有监测数据，则用 0 填充所有统计特征 (重要：处理空窗口)
            for feature_name in feature_columns:
                for stat_func in STAT_FUNCTIONS:
                    new_col_name = f"{feature_name}_{stat_func}"
                    # 使用 0 填充 (或 np.nan，取决于后续处理，这里保持使用 0)
                    data_row[new_col_name] = 0.0 
        
        final_data_list.append(data_row)

    final_df = pd.DataFrame(final_data_list)
    print(f"  -> 最终特征维度: {len(feature_columns) * len(STAT_FUNCTIONS)} 维 (44 * 5 = 220)")
    print(f"  -> 最终数据集共 {len(final_df)} 个样本。")
    return final_df

    
# def feature_engineering_and_padding(monitor_df, target_lookup_df):
#     """进行特征收集、展平、零填充，并构建最终数据集。"""
#     print("--- 2. 特征工程: 展平和填充 ---")
    
#     feature_columns = [col for col in monitor_df.columns if col.startswith('feature')]
#     num_features_per_record = len(feature_columns)
    
#     max_records = 0
#     window_monitor_data = {} 

#     # 第一次循环：确定 N_max
#     for index, row in target_lookup_df.iterrows():
#         start_time = row['开始时间']
#         end_time = row['结束时间']
#         interval_data = monitor_df[
#             (monitor_df['时间'] >= start_time) & (monitor_df['时间'] < end_time)
#         ]
#         current_records = len(interval_data)
#         if current_records > 0:
#             max_records = max(max_records, current_records)
#             window_monitor_data[index] = interval_data[feature_columns].values.flatten()
#         else:
#             window_monitor_data[index] = np.array([])
            
#     if max_records == 0:
#         raise RuntimeError("致命错误：没有找到任何包含监测数据的时间窗口。")

#     total_features = max_records * num_features_per_record
#     print(f"  -> N_max (最大记录数): {max_records} 条")
#     print(f"  -> 每个样本总特征数: {total_features}")
    
#     final_data_list = []
#     # 创建特征列名 (F{Feature}_R{Record})
#     feature_names = [f'F{i+1}_R{r+1}' for r in range(max_records) for i in range(num_features_per_record)]

#     # 第二次循环：构建最终数据集，进行零填充
#     for index, row in target_lookup_df.iterrows():
#         delta_x = row['Delta_X']
#         delta_y = row['Delta_Y']
        
#         flattened_features = window_monitor_data.get(index, np.array([]))
        
#         # 零填充 (Padding)
#         padding_needed = total_features - len(flattened_features)
#         padded_features = np.pad(flattened_features, (0, padding_needed), 'constant', constant_values=0)
        
#         data_row = {'Delta_X': delta_x, 'Delta_Y': delta_y}
#         data_row.update(zip(feature_names, padded_features))
#         final_data_list.append(data_row)

#     final_df = pd.DataFrame(final_data_list)
#     print(f"  -> 最终数据集共 {len(final_df)} 个样本。")
#     return final_df

def split_and_save_data(final_df, test_size=0.2):
    """按时间顺序划分数据集并保存为 CSV 文件。"""
    print("--- 3. 数据划分和保存 ---")
    
    X = final_df.drop(columns=['Delta_X', 'Delta_Y'])
    y = final_df[['Delta_X', 'Delta_Y']]
    
    # 按时间顺序划分
    split_point = int(len(final_df) * (1 - test_size))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # X_train = X.iloc[:split_point]
    # X_test = X.iloc[split_point:]
    # y_train = y.iloc[:split_point]
    # y_test = y.iloc[split_point:]

    # 保存文件
    X_train_file = os.path.join(PROCESSED_DATA_PATH, 'X_train_static_random.csv')
    X_test_file = os.path.join(PROCESSED_DATA_PATH, 'X_test_static_random.csv')
    y_train_file = os.path.join(PROCESSED_DATA_PATH, 'y_train_static_random.csv')
    y_test_file = os.path.join(PROCESSED_DATA_PATH, 'y_test_static_random.csv')
    
    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)
    
    print(f"  -> 训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}")
    print(f"  -> 数据集已保存至 {PROCESSED_DATA_PATH} 目录。")
    
if __name__ == '__main__':
    # 主流程
    monitor_df, target_lookup_df = load_and_preprocess_raw_data()
    final_df = feature_engineering_and_padding(monitor_df, target_lookup_df)
    split_and_save_data(final_df)