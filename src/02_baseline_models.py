# src/02_baseline_models.py

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from utils import evaluate_model_performance 

# 定义文件路径
PROCESSED_DATA_PATH = 'data/processed'
X_train_file = os.path.join(PROCESSED_DATA_PATH, 'X_train.csv')
X_test_file = os.path.join(PROCESSED_DATA_PATH, 'X_test.csv')
y_train_file = os.path.join(PROCESSED_DATA_PATH, 'y_train.csv')
y_test_file = os.path.join(PROCESSED_DATA_PATH, 'y_test.csv')

def load_processed_data():
    """从 CSV 文件加载训练和测试数据。"""
    print("--- 1. 加载处理后的数据 ---")
    try:
        X_train = pd.read_csv(X_train_file)
        X_test = pd.read_csv(X_test_file)
        y_train = pd.read_csv(y_train_file)
        y_test = pd.read_csv(y_test_file)
        print("  -> 数据加载成功。")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("错误：找不到处理后的数据。请先运行 src/01_data_prep.py！")
        raise

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name, use_scaler=False):
    """训练并评估模型。"""
    print(f"\n--- 2. 训练模型: {name} ---")
    
    X_train_data = X_train
    X_test_data = X_test
    
    # 如果需要标准化 (适用于线性模型、MLP)
    if use_scaler:
        scaler = StandardScaler()
        X_train_data = scaler.fit_transform(X_train)
        X_test_data = scaler.transform(X_test)
        
    model.fit(X_train_data, y_train)
    y_pred = model.predict(X_test_data)
    
    # 使用工具函数评估
    results = evaluate_model_performance(y_test, y_pred, name)
    return results

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_processed_data()
    
    all_results = {}
    
    # 线性回归 (需要标准化)
    lr_model = LinearRegression()
    lr_results = train_and_evaluate_model(
        lr_model, X_train, X_test, y_train, y_test, "Linear Regression", use_scaler=True
    )
    all_results["LR"] = lr_results

    # 随机森林 (无需标准化)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_results = train_and_evaluate_model(
        rf_model, X_train, X_test, y_train, y_test, "Random Forest", use_scaler=False
    )
    all_results["RF"] = rf_results

    # XGBoost 回归 (无需标准化, 使用 MultiOutputRegressor 封装)
    xgb_base = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model = MultiOutputRegressor(xgb_base)
    xgb_results = train_and_evaluate_model(
        xgb_model, X_train, X_test, y_train, y_test, "XGBoost", use_scaler=False
    )
    all_results["XGB"] = xgb_results
    
    print("\n\n=============== 最终性能总结 ===============")
    print(pd.DataFrame(all_results).T[['R2_weighted', 'MSE']].sort_values(by='R2_weighted', ascending=False))