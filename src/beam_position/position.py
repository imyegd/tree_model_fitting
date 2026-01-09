import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 加载数据
beam_monitor_df = pd.read_csv("data\束位监测数据.csv")
beam_position_df = pd.read_csv("data\束位数据.csv")

def prepare_time_aligned_targets(monitor_df, position_df):
    """
    处理时间转换、日期推断、计算 Delta X/Y，并创建时间窗口 [开始时间, 结束时间) 的查找表。
    """
    # 1. Monitor Data Time Prep
    monitor_df['时间'] = pd.to_datetime(monitor_df['时间'])
    monitor_df = monitor_df.sort_values(by='时间').reset_index(drop=True)

    # 2. Position Data Time Prep (日期推断)
    position_df['时间_only'] = pd.to_datetime(
        position_df['时间'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    
    start_date_monitor = monitor_df['时间'].min().date()

    def reconstruct_datetime(df_time, base_date):
        datetimes = []
        current_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[0]))
        datetimes.append(current_datetime)

        for i in range(1, len(df_time)):
            next_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[i]))
            # 检查跨天情况
            if next_datetime <= current_datetime:
                base_date = base_date + timedelta(days=1)
                next_datetime = pd.to_datetime(str(base_date) + ' ' + str(df_time.iloc[i]))
            
            datetimes.append(next_datetime)
            current_datetime = next_datetime
        return pd.Series(datetimes)

    full_datetimes = reconstruct_datetime(position_df['时间_only'], start_date_monitor)
    position_df['完整时间'] = full_datetimes
    position_df = position_df.sort_values(by='完整时间').reset_index(drop=True)

    # 3. 计算目标 (Delta X, Delta Y)
    # 目标：当前坐标 - 前一时刻坐标
    position_df['Delta_X'] = position_df['束位X'].diff()
    position_df['Delta_Y'] = position_df['束位Y'].diff()
    position_df.dropna(subset=['Delta_X', 'Delta_Y'], inplace=True)
    position_df.reset_index(drop=True, inplace=True)

    # 4. 创建时间窗口查找表
    target_lookup_df = position_df[['完整时间', 'Delta_X', 'Delta_Y']].copy()
    target_lookup_df['开始时间'] = target_lookup_df['完整时间'].shift(1)
    target_lookup_df.dropna(subset=['开始时间'], inplace=True)
    target_lookup_df.rename(columns={'完整时间': '结束时间'}, inplace=True)
    target_lookup_df.reset_index(drop=True, inplace=True)

    return monitor_df, target_lookup_df

# 调用预处理函数
beam_monitor_df, target_lookup_df = prepare_time_aligned_targets(beam_monitor_df, beam_position_df)

# 识别特征列
feature_columns = [col for col in beam_monitor_df.columns if col.startswith('feature')]
num_features_per_record = len(feature_columns)

# --- 3. 特征收集、展平和填充 (Flattening and Padding) ---

# 3.1. 确定最大的记录数 N_max
max_records = 0
window_monitor_data = {} # 存储每个窗口的原始展平数据

print("步骤 1: 计算每个时间窗口内的最大监测数据记录数...")
for index, row in target_lookup_df.iterrows():
    start_time = row['开始时间']
    end_time = row['结束时间']
    
    # 过滤出当前时间间隔内的监测数据: [开始时间, 结束时间)
    interval_data = beam_monitor_df[
        (beam_monitor_df['时间'] >= start_time) & (beam_monitor_df['时间'] < end_time)
    ]
    
    current_records = len(interval_data)
    if current_records > 0:
        max_records = max(max_records, current_records)
        # 展平特征数据
        window_monitor_data[index] = interval_data[feature_columns].values.flatten()
    else:
        # 如果窗口内没有监测数据，我们仍然保留这个样本，但数据用一个空数组表示，稍后会全部填充0
        window_monitor_data[index] = np.array([])
        # ⚠️ 注意：如果跳过，会导致时间序列不连续，所以我们保留它并用 0 填充。

if max_records == 0:
    print("⚠️ 致命错误：没有找到任何包含监测数据的时间窗口。请检查时间戳对齐问题。")
    exit()

# 计算每个样本最终的特征总数 M_total
total_features = max_records * num_features_per_record
print(f"步骤 2: 最大监测数据记录数 (N_max): {max_records} 条")
print(f"每个样本的最终特征总数: {total_features} ( {max_records} records x {num_features_per_record} features)")

# 3.2. 构建最终数据集
final_data_list = []

# 生成特征列名 (例如: F_1_R_1, F_2_R_1, ..., F_M_R_1, F_1_R_2, ...)
feature_names = [f'F{i+1}_R{r+1}' for r in range(max_records) for i in range(num_features_per_record)]

for index, row in target_lookup_df.iterrows():
    delta_x = row['Delta_X']
    delta_y = row['Delta_Y']
    
    flattened_features = window_monitor_data.get(index, np.array([]))
    current_features_len = len(flattened_features)
    
    # 填充 (Pad)
    if current_features_len < total_features:
        # 使用零进行填充。如果原始特征是连续的，零填充可以理解为“没有信号”
        padding_needed = total_features - current_features_len
        padded_features = np.pad(flattened_features, (0, padding_needed), 'constant', constant_values=0)
    else:
        padded_features = flattened_features
    
    # 构建数据行
    data_row = {
        'Delta_X': delta_x,
        'Delta_Y': delta_y
    }
    
    # 将展平的特征添加到字典中
    # 使用 zip 确保特征名称和值一一对应
    data_row.update(zip(feature_names, padded_features))

    final_data_list.append(data_row)

final_df = pd.DataFrame(final_data_list)

# 4. 划分数据集 (时间序列)

if final_df.empty:
    print("⚠️ 警告：最终数据集为空。无法进行划分。")
else:
    print(f"\n✅ 成功构建最终数据集，共 {len(final_df)} 个样本。")
    
    # 特征 (X) 和 目标 (y)
    X = final_df.drop(columns=['Delta_X', 'Delta_Y'])
    y = final_df[['Delta_X', 'Delta_Y']]
    
    # 划分点：前 80% 作为训练集 (保持时间顺序)
    split_point = int(len(final_df) * 0.8)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print("\n--- 数据集划分结果 ---")
    print(f"训练集大小: {len(X_train)} (约 {split_point / len(final_df) * 100:.1f}%)")
    print(f"测试集大小: {len(X_test)} (约 {(len(final_df) - split_point) / len(final_df) * 100:.1f}%)")
    print(f"每个样本的特征总数: {X_train.shape[1]}")
    
    print("\n训练集特征 (X_train) 摘要 (前5行):")
    print(X_train.head())
    print("\n训练集目标 (y_train) 摘要 (前5行):")
    print(y_train.head())

# 定义文件名
X_train_file = 'data\X_train.csv'
X_test_file = 'data\X_test.csv'
y_train_file = 'data\y_train.csv'
y_test_file = 'data\y_test.csv'

# 保存特征集 (X)
X_train.to_csv(X_train_file, index=False)
X_test.to_csv(X_test_file, index=False)

# 保存目标集 (y)
y_train.to_csv(y_train_file, index=False)
y_test.to_csv(y_test_file, index=False)

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # 假设 X_train, X_test, y_train, y_test 已经从上一步的代码中获得

# # 1. 数据预处理：标准化 (Standardization)
# # 线性模型和神经网络对特征尺度敏感，因此进行标准化是必要的。
# scaler = StandardScaler()

# # 对训练数据拟合scaler，并进行转换
# X_train_scaled = scaler.fit_transform(X_train)
# # 对测试数据只进行转换
# X_test_scaled = scaler.transform(X_test)

# print("--- 开始模型训练和评估 ---")

# results = {}

# # --- 2. 线性回归 (Linear Regression) ---
# # 线性回归本身支持多输出
# print("\n[模型 1: 线性回归]")
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)
# y_pred_lr = lr_model.predict(X_test_scaled)

# # 评估
# r2_lr = r2_score(y_test, y_pred_lr, multioutput='variance_weighted')
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# results['Linear Regression'] = {'R2': r2_lr, 'MSE': mse_lr}

# print(f"R-squared (加权平均): {r2_lr:.4f}")
# print(f"Mean Squared Error: {mse_lr:.4f}")


# # --- 3. 随机森林回归 (Random Forest Regressor) ---
# # 使用 MultiOutputRegressor 包装，因为它对多输出的优化不如线性模型直接
# print("\n[模型 2: 随机森林回归]")
# rf_base_model = RandomForestRegressor(
#     n_estimators=100,      # 树的数量
#     max_depth=10,          # 树的最大深度，限制过拟合
#     random_state=42, 
#     n_jobs=-1              # 使用所有核心进行并行计算
# )
# # 随机森林本身支持多输出，但使用 MultiOutputRegressor 是一种标准做法
# rf_model = rf_base_model
# rf_model.fit(X_train, y_train) # 随机森林对尺度不敏感，使用原始 X_train
# y_pred_rf = rf_model.predict(X_test)

# # 评估
# r2_rf = r2_score(y_test, y_pred_rf, multioutput='variance_weighted')
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# results['Random Forest'] = {'R2': r2_rf, 'MSE': mse_rf}

# print(f"R-squared (加权平均): {r2_rf:.4f}")
# print(f"Mean Squared Error: {mse_rf:.4f}")


# # --- 4. 多层感知器 / 神经网络 (MLP Regressor) ---
# print("\n[模型 3: 简单 MLP 神经网络]")
# # 默认的 MLP 只支持单输出，所以必须使用 MultiOutputRegressor 包装
# mlp_base_model = MLPRegressor(
#     hidden_layer_sizes=(64, 32), # 两个隐藏层，分别有64和32个神经元
#     activation='relu',           # 激活函数
#     solver='adam',               # 优化器
#     max_iter=500,                # 最大迭代次数
#     random_state=42,
#     early_stopping=True          # 启用早停防止过拟合
# )
# mlp_model = MultiOutputRegressor(mlp_base_model)
# mlp_model.fit(X_train_scaled, y_train)
# y_pred_mlp = mlp_model.predict(X_test_scaled)

# # 评估
# r2_mlp = r2_score(y_test, y_pred_mlp, multioutput='variance_weighted')
# mse_mlp = mean_squared_error(y_test, y_pred_mlp)
# results['MLP Regressor'] = {'R2': r2_mlp, 'MSE': mse_mlp}

# print(f"R-squared (加权平均): {r2_mlp:.4f}")
# print(f"Mean Squared Error: {mse_mlp:.4f}")


# # --- 5. 结果总结 ---
# print("\n--- 🚀 模型性能总结 (在测试集上) 🚀 ---")
# summary_df = pd.DataFrame(results).T
# print(summary_df.sort_values(by='R2', ascending=False))