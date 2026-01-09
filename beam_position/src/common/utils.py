# src/common/utils.py

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. 时间处理工具 ---

def reconstruct_datetime(df_time, base_date):
    """
    根据时间 (H:M:S) 和起始日期，推断完整的 datetime 对象，处理跨天情况。
    
    参数:
        df_time (pd.Series): 只有时间部分的 Series (pd.time对象)。
        base_date (pd.date): 用于起始推断的日期。
    返回:
        pd.Series: 完整的 datetime 对象 Series。
    """
    datetimes = []
    # 初始化第一个完整时间
    current_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[0]))
    datetimes.append(current_datetime)

    for i in range(1, len(df_time)):
        next_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[i]))
        # 如果下一时间早于或等于当前时间，则说明跨天了
        if next_datetime <= current_datetime:
            base_date = base_date + timedelta(days=1)
            next_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[i]))
        
        datetimes.append(next_datetime)
        current_datetime = next_datetime
        
    return pd.Series(datetimes)

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    计算并打印多输出回归模型的性能指标 (R2, MSE, Delta_X/Y R2)。
    
    参数:
        y_true (np.array/pd.DataFrame): 真实值。
        y_pred (np.array/pd.DataFrame): 预测值。
        model_name (str): 模型名称，用于打印输出。
    """
    # 确保输入是 NumPy 数组
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # 整体加权平均 R2
    r2_weighted = r2_score(y_true, y_pred, multioutput='variance_weighted')
    # 整体均方误差
    mse = mean_squared_error(y_true, y_pred)

    # 分别计算 Delta_X 和 Delta_Y 的 R2 (假设 Delta_X是第0列，Delta_Y是第1列)
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])

    print(f"\n--- 评估结果: {model_name} ---")
    print(f"R-squared (加权平均): {r2_weighted:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Delta_X (束位X差值) R-squared: {r2_x:.4f}")
    print(f"Delta_Y (束位Y差值) R-squared: {r2_y:.4f}")
    
    return {'R2_weighted': r2_weighted, 'MSE': mse, 'R2_X': r2_x, 'R2_Y': r2_y}

# --- 2. 数据重塑工具 ---

def reshape_for_lstm(X_data, num_original_features):
    """
    将展平的特征数据重塑为 LSTM 所需的 3D 格式。
    
    形状: [样本数, N_max * F] -> [样本数, N_max, F]
    
    参数:
        X_data (np.array/pd.DataFrame): 展平后的特征数据。
        num_original_features (int): 每个原始监测记录的特征数量 (F)。
    返回:
        np.array: 重塑后的 3D 数组。
    """
    if isinstance(X_data, pd.DataFrame):
        X_data = X_data.values
        
    num_samples = X_data.shape[0]
    num_total_features = X_data.shape[1]
    
    if num_total_features % num_original_features != 0:
        raise ValueError("总特征数不能被原始特征数整除，请检查数据展平逻辑。")
        
    time_steps = num_total_features // num_original_features
    
    return X_data.reshape(num_samples, time_steps, num_original_features)
