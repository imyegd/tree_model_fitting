# src/02_baseline_models.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.common.utils import evaluate_model_performance 

# 定义文件路径
PROCESSED_DATA_PATH = './data/processed'

def load_processed_data(data_source='static_timesplit'):
    """
    从 CSV 文件加载训练和测试数据。
    
    参数:
        data_source: 数据来源标识，例如 'static_randomsplit', 'static_timesplit', 
                    'raw_randomsplit', 'raw_timesplit'
    """
    print(f"--- 1. 加载处理后的数据 ({data_source}) ---")
    
    X_train_file = os.path.join(PROCESSED_DATA_PATH, f'X_train_{data_source}.csv')
    X_test_file = os.path.join(PROCESSED_DATA_PATH, f'X_test_{data_source}.csv')
    y_train_file = os.path.join(PROCESSED_DATA_PATH, f'y_train_{data_source}.csv')
    y_test_file = os.path.join(PROCESSED_DATA_PATH, f'y_test_{data_source}.csv')
    
    try:
        X_train = pd.read_csv(X_train_file)
        X_test = pd.read_csv(X_test_file)
        y_train = pd.read_csv(y_train_file)
        y_test = pd.read_csv(y_test_file)
        print("  -> 数据加载成功。")
        return X_train, X_test, y_train, y_test, data_source
    except FileNotFoundError:
        print(f"错误：找不到处理后的数据 ({data_source})。请先运行 src/01_data_prep.py！")
        raise


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name, data_source, use_scaler=False):
    """
    训练并评估模型。
    
    参数:
        model: 模型对象
        X_train, X_test, y_train, y_test: 训练和测试数据
        name: 模型名称
        data_source: 数据来源标识，用于组织保存路径
        use_scaler: 是否使用标准化
    """
    print(f"\n--- 2. 训练模型: {name} ---")
    
    X_train_data = X_train
    X_test_data = X_test
    
    # 如果需要标准化 (适用于线性模型、MLP)
    if use_scaler:
        scaler = StandardScaler()
        X_train_data = scaler.fit_transform(X_train)
        X_test_data = scaler.transform(X_test)
        
    model.fit(X_train_data, y_train)
    
    # 对训练集和测试集都进行预测
    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)
    
    # 使用工具函数评估（只评估测试集）
    results = evaluate_model_performance(y_test, y_test_pred, name)
    
    # 绘制训练集和测试集的实际值和预测值的对比图
    plot_prediction_comparison(y_train, y_train_pred, y_test, y_test_pred, name, data_source)
    
    return results


def plot_prediction_comparison(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name, data_source):
    """
    绘制训练集和测试集的实际XY差分和预测XY差分的对比图，以及还原的原始坐标对比
    
    Args:
        y_train_true: 训练集真实值 (DataFrame 或 numpy array)，包含 Delta_X 和 Delta_Y
        y_train_pred: 训练集预测值 (numpy array)，包含 Delta_X 和 Delta_Y
        y_test_true: 测试集真实值 (DataFrame 或 numpy array)，包含 Delta_X 和 Delta_Y
        y_test_pred: 测试集预测值 (numpy array)，包含 Delta_X 和 Delta_Y
        model_name: 模型名称
        data_source: 数据来源标识，用于组织保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import os
    
    # 确保是 numpy 数组
    if isinstance(y_train_true, pd.DataFrame):
        y_train_true = y_train_true.values
    if isinstance(y_train_pred, pd.DataFrame):
        y_train_pred = y_train_pred.values
    if isinstance(y_test_true, pd.DataFrame):
        y_test_true = y_test_true.values
    if isinstance(y_test_pred, pd.DataFrame):
        y_test_pred = y_test_pred.values
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表 - 3行3列布局
    fig = plt.figure(figsize=(24, 18))
    
    # 提取训练集的 Delta_X 和 Delta_Y
    train_true_delta_x = y_train_true[:, 0]
    train_true_delta_y = y_train_true[:, 1]
    train_pred_delta_x = y_train_pred[:, 0]
    train_pred_delta_y = y_train_pred[:, 1]
    
    # 提取测试集的 Delta_X 和 Delta_Y
    test_true_delta_x = y_test_true[:, 0]
    test_true_delta_y = y_test_true[:, 1]
    test_pred_delta_x = y_test_pred[:, 0]
    test_pred_delta_y = y_test_pred[:, 1]
    
    # 从差分还原原始坐标（累积求和）
    # 训练集
    train_true_X_original = np.cumsum(train_true_delta_x)
    train_true_Y_original = np.cumsum(train_true_delta_y)
    train_pred_X_original = np.cumsum(train_pred_delta_x)
    train_pred_Y_original = np.cumsum(train_pred_delta_y)
    
    # 测试集（需要从训练集的最后一个值开始累积）
    train_end_X = train_true_X_original[-1] if len(train_true_X_original) > 0 else 0
    train_end_Y = train_true_Y_original[-1] if len(train_true_Y_original) > 0 else 0
    test_true_X_original = train_end_X + np.cumsum(test_true_delta_x)
    test_true_Y_original = train_end_Y + np.cumsum(test_true_delta_y)
    test_pred_X_original = train_end_X + np.cumsum(test_pred_delta_x)
    test_pred_Y_original = train_end_Y + np.cumsum(test_pred_delta_y)
    
    # 1. (Delta_X, Delta_Y) 2D坐标对比图
    ax_main = plt.subplot(3, 3, 1)
    # 训练集
    ax_main.scatter(train_true_delta_x, train_true_delta_y, alpha=0.5, s=10, c='lightblue', 
                   label='训练集-实际 (Delta_X, Delta_Y)', marker='o', edgecolors='blue', linewidths=0.5)
    ax_main.scatter(train_pred_delta_x, train_pred_delta_y, alpha=0.5, s=10, c='lightcoral', 
                   label='训练集-预测 (Delta_X, Delta_Y)', marker='s', edgecolors='red', linewidths=0.5)
    # 测试集
    ax_main.scatter(test_true_delta_x, test_true_delta_y, alpha=0.7, s=15, c='blue', 
                   label='测试集-实际 (Delta_X, Delta_Y)', marker='o')
    ax_main.scatter(test_pred_delta_x, test_pred_delta_y, alpha=0.7, s=15, c='red', 
                   label='测试集-预测 (Delta_X, Delta_Y)', marker='s')
    ax_main.set_title(f'{model_name} - (Delta_X, Delta_Y) 坐标对比', 
                     fontsize=12, fontweight='bold')
    ax_main.set_xlabel('Delta_X', fontsize=11)
    ax_main.set_ylabel('Delta_Y', fontsize=11)
    ax_main.legend(loc='best', fontsize=8)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_main.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. 原始束位坐标 (X, Y) 对比图
    ax_position = plt.subplot(3, 3, 2)
    # 训练集
    ax_position.scatter(train_true_X_original, train_true_Y_original, alpha=0.5, s=10, 
                       c='lightblue', label='训练集-实际 (束位X, 束位Y)', marker='o', 
                       edgecolors='blue', linewidths=0.5)
    ax_position.scatter(train_pred_X_original, train_pred_Y_original, alpha=0.5, s=10, 
                       c='lightcoral', label='训练集-预测 (束位X, 束位Y)', marker='s', 
                       edgecolors='red', linewidths=0.5)
    # 测试集
    ax_position.scatter(test_true_X_original, test_true_Y_original, alpha=0.7, s=15, 
                       c='blue', label='测试集-实际 (束位X, 束位Y)', marker='o')
    ax_position.scatter(test_pred_X_original, test_pred_Y_original, alpha=0.7, s=15, 
                       c='red', label='测试集-预测 (束位X, 束位Y)', marker='s')
    ax_position.set_title(f'{model_name} - 原始束位坐标 (X, Y) 对比', 
                         fontsize=12, fontweight='bold')
    ax_position.set_xlabel('束位X', fontsize=11)
    ax_position.set_ylabel('束位Y', fontsize=11)
    ax_position.legend(loc='best', fontsize=9)
    ax_position.grid(True, alpha=0.3)
    
    # 3. Delta_X 时间序列对比（训练集+测试集）
    ax1 = plt.subplot(3, 3, 4)
    train_indices = np.arange(len(train_true_delta_x))
    test_indices = np.arange(len(train_true_delta_x), len(train_true_delta_x) + len(test_true_delta_x))
    # 训练集
    ax1.plot(train_indices, train_true_delta_x, 'b-', linewidth=1.0, alpha=0.5, label='训练集-实际 Delta_X')
    ax1.plot(train_indices, train_pred_delta_x, 'r--', linewidth=1.0, alpha=0.5, label='训练集-预测 Delta_X')
    # 测试集
    ax1.plot(test_indices, test_true_delta_x, 'b-', linewidth=1.5, alpha=0.8, label='测试集-实际 Delta_X')
    ax1.plot(test_indices, test_pred_delta_x, 'r--', linewidth=1.5, alpha=0.8, label='测试集-预测 Delta_X')
    # 添加分割线
    ax1.axvline(x=len(train_true_delta_x), color='green', linestyle=':', linewidth=2, 
                alpha=0.7, label='训练/测试分界')
    ax1.set_title(f'{model_name} - Delta_X 对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('样本序号')
    ax1.set_ylabel('Delta_X 值')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 4. Delta_Y 时间序列对比（训练集+测试集）
    ax2 = plt.subplot(3, 3, 5)
    # 训练集
    ax2.plot(train_indices, train_true_delta_y, 'b-', linewidth=1.0, alpha=0.5, label='训练集-实际 Delta_Y')
    ax2.plot(train_indices, train_pred_delta_y, 'r--', linewidth=1.0, alpha=0.5, label='训练集-预测 Delta_Y')
    # 测试集
    ax2.plot(test_indices, test_true_delta_y, 'b-', linewidth=1.5, alpha=0.8, label='测试集-实际 Delta_Y')
    ax2.plot(test_indices, test_pred_delta_y, 'r--', linewidth=1.5, alpha=0.8, label='测试集-预测 Delta_Y')
    # 添加分割线
    ax2.axvline(x=len(train_true_delta_x), color='green', linestyle=':', linewidth=2, 
                alpha=0.7, label='训练/测试分界')
    ax2.set_title(f'{model_name} - Delta_Y 对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('样本序号')
    ax2.set_ylabel('Delta_Y 值')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 5. 原始束位X时间序列对比（训练集+测试集）
    ax3 = plt.subplot(3, 3, 7)
    # 训练集
    ax3.plot(train_indices, train_true_X_original, 'b-', linewidth=1.0, alpha=0.5, label='训练集-实际 束位X')
    ax3.plot(train_indices, train_pred_X_original, 'r--', linewidth=1.0, alpha=0.5, label='训练集-预测 束位X')
    # 测试集
    ax3.plot(test_indices, test_true_X_original, 'b-', linewidth=1.5, alpha=0.8, label='测试集-实际 束位X')
    ax3.plot(test_indices, test_pred_X_original, 'r--', linewidth=1.5, alpha=0.8, label='测试集-预测 束位X')
    # 添加分割线
    ax3.axvline(x=len(train_true_delta_x), color='green', linestyle=':', linewidth=2, 
                alpha=0.7, label='训练/测试分界')
    ax3.set_title(f'{model_name} - 束位X 对比', fontsize=12, fontweight='bold')
    ax3.set_xlabel('样本序号')
    ax3.set_ylabel('束位X 值')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 6. 原始束位Y时间序列对比（训练集+测试集）【新增】
    ax4 = plt.subplot(3, 3, 8)
    # 训练集
    ax4.plot(train_indices, train_true_Y_original, 'b-', linewidth=1.0, alpha=0.5, label='训练集-实际 束位Y')
    ax4.plot(train_indices, train_pred_Y_original, 'r--', linewidth=1.0, alpha=0.5, label='训练集-预测 束位Y')
    # 测试集
    ax4.plot(test_indices, test_true_Y_original, 'b-', linewidth=1.5, alpha=0.8, label='测试集-实际 束位Y')
    ax4.plot(test_indices, test_pred_Y_original, 'r--', linewidth=1.5, alpha=0.8, label='测试集-预测 束位Y')
    # 添加分割线
    ax4.axvline(x=len(train_true_delta_x), color='green', linestyle=':', linewidth=2, 
                alpha=0.7, label='训练/测试分界')
    ax4.set_title(f'{model_name} - 束位Y 对比', fontsize=12, fontweight='bold')
    ax4.set_xlabel('样本序号')
    ax4.set_ylabel('束位Y 值')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - 训练集与测试集实际与预测对比图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片 - 根据数据来源创建子目录
    result_dir = os.path.join('./result/baseline_models', data_source)
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(result_dir, f"{timestamp}_{model_name.replace(' ', '_')}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存为: {plot_path}")
    
    plt.show()

if __name__ == '__main__':
    # 可以修改这里来切换不同的数据来源
    data_source = 'raw_timesplit'  # 可选: 'static_randomsplit', 'static_timesplit', 'raw_randomsplit', 'raw_timesplit'
    
    X_train, X_test, y_train, y_test, _ = load_processed_data(data_source)
    
    all_results = {}
    
    # 线性回归 (需要标准化)
    lr_model = LinearRegression()
    lr_results = train_and_evaluate_model(
        lr_model, X_train, X_test, y_train, y_test, "Linear Regression", data_source, use_scaler=True
    )
    all_results["LR"] = lr_results

    # 随机森林 (无需标准化)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_results = train_and_evaluate_model(
        rf_model, X_train, X_test, y_train, y_test, "Random Forest", data_source, use_scaler=False
    )
    all_results["RF"] = rf_results

    # XGBoost 回归 (无需标准化, 使用 MultiOutputRegressor 封装)
    xgb_base = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model = MultiOutputRegressor(xgb_base)
    xgb_results = train_and_evaluate_model(
        xgb_model, X_train, X_test, y_train, y_test, "XGBoost", data_source, use_scaler=False
    )
    all_results["XGB"] = xgb_results
    
    # 决策树 (无需标准化, 使用 MultiOutputRegressor 封装)
    dt_base = DecisionTreeRegressor(max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
    dt_model = MultiOutputRegressor(dt_base)
    dt_results = train_and_evaluate_model(
        dt_model, X_train, X_test, y_train, y_test, "Decision Tree", data_source, use_scaler=False
    )
    all_results["DT"] = dt_results
    
    # 梯度提升树 (无需标准化, 使用 MultiOutputRegressor 封装)
    gb_base = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)
    gb_model = MultiOutputRegressor(gb_base)
    gb_results = train_and_evaluate_model(
        gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting", data_source, use_scaler=False
    )
    all_results["GB"] = gb_results
    
    # MLP (多层感知机, 需要标准化)
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        activation='relu', 
        solver='adam', 
        alpha=0.001,
        batch_size='auto', 
        learning_rate='constant', 
        learning_rate_init=0.001,
        max_iter=500, 
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    mlp_results = train_and_evaluate_model(
        mlp_model, X_train, X_test, y_train, y_test, "MLP", data_source, use_scaler=True
    )
    all_results["MLP"] = mlp_results
    
    print(f"\n\n=============== 最终性能总结 ({data_source}) ===============")
    print(pd.DataFrame(all_results).T[['R2_weighted', 'MSE']].sort_values(by='R2_weighted', ascending=False))