#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用树模型拟合束流数据
支持决策树、随机森林、XGBoost等树模型
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import BaseCrossValidator
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置字体回退，让matplotlib自动选择可用字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 尝试导入XGBoost，如果不可用则跳过
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")

def ensure_result_dir():
    """确保result目录存在"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./result/束流/tree_model/{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果目录: {result_dir}")
    return result_dir

def load_and_prepare_data(csv_file_path, n_samples=None):
    """
    加载和准备数据
    
    Args:
        csv_file_path (str): CSV文件路径
        n_samples (int): 使用的样本数量，None表示使用所有数据
    
    Returns:
        tuple: (X, y) 特征矩阵和目标变量
    """
    # 读取CSV文件
    print(f"正在读取数据文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 选择数据
    if n_samples is None:
        df_subset = df
        print(f"使用所有 {df.shape[0]} 条数据")
    else:
        df_subset = df.head(n_samples)
        print(f"使用前{n_samples}条数据")
    
    # 提取特征列（所有以feature开头的列）
    feature_columns = [col for col in df_subset.columns if col.startswith('feature')]
    
    # exclude_features = ['feature4']
    # if exclude_features:
    #     original_count = len(feature_columns)
    #     feature_columns = [col for col in feature_columns if col not in exclude_features]
    #     excluded = set(exclude_features) & set(feature_columns + exclude_features)
    #     print(f"排除了特征: {excluded}")
    #     print(f"特征数量: {original_count} -> {len(feature_columns)}")
    print(f"特征列: {feature_columns}")
    
    # 提取目标变量
    if 'target' in df_subset.columns:
        target_column = 'target'
    elif '束流' in df_subset.columns:
        target_column = '束流'
    else:
        raise ValueError("未找到目标变量列")
    
    print(f"目标变量列: {target_column}")
    
    # 准备特征矩阵和目标变量
    X = df_subset[feature_columns].values
    y = df_subset[target_column].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    return X, y, feature_columns

# def split_data(X, y, test_size=0.2, random_state=42):
#     """
#     划分训练集和测试集
    
#     Args:
#         X (numpy.ndarray): 特征矩阵
#         y (numpy.ndarray): 目标变量
#         test_size (float): 测试集比例
#         random_state (int): 随机种子
    
#     Returns:
#         tuple: (X_train, X_test, y_train, y_test)
#     """
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )
    
#     print(f"训练集大小: {X_train.shape[0]} 样本")
#     print(f"测试集大小: {X_test.shape[0]} 样本")
#     print(f"训练集特征维度: {X_train.shape[1]}")
#     print(f"测试集特征维度: {X_test.shape[1]}")
    
#     return X_train, X_test, y_train, y_test

def split_data(X, y, test_size=0.2, random_state=None):
    """
    划分训练集和测试集（按顺序划分，不使用随机）
    
    Args:
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        test_size (float): 测试集比例
        random_state: 该参数已废弃，保留仅为兼容性
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # 计算分割点
    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - test_size))
    
    # 按顺序划分：前80%为训练集，后20%为测试集
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    print(f"训练集特征维度: {X_train.shape[1]}")
    print(f"测试集特征维度: {X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    训练决策树模型
    
    Args:
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
        max_depth (int): 最大深度
        min_samples_split (int): 内部节点分裂所需的最小样本数
        min_samples_leaf (int): 叶节点所需的最小样本数
    
    Returns:
        DecisionTreeRegressor: 训练好的模型
    """
    print("开始训练决策树模型...")
    
    # 创建决策树模型
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("决策树模型训练完成")
    print(f"模型深度: {model.get_depth()}")
    print(f"叶节点数: {model.get_n_leaves()}")
    
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    训练随机森林模型
    
    Args:
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
        n_estimators (int): 树的数量
        max_depth (int): 最大深度
        min_samples_split (int): 内部节点分裂所需的最小样本数
        min_samples_leaf (int): 叶节点所需的最小样本数
        random_state (int): 随机种子
    
    Returns:
        RandomForestRegressor: 训练好的模型
    """
    print("开始训练随机森林模型...")
    
    # 创建随机森林模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1  # 使用所有可用CPU核心
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("随机森林模型训练完成")
    print(f"树的数量: {model.n_estimators}")
    print(f"特征重要性总和: {model.feature_importances_.sum():.6f}")
    
    return model

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
    """
    训练XGBoost模型
    
    Args:
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
        n_estimators (int): 树的数量
        max_depth (int): 最大深度
        learning_rate (float): 学习率
        subsample (float): 子样本比例
        colsample_bytree (float): 特征采样比例
        random_state (int): 随机种子
    
    Returns:
        XGBRegressor: 训练好的模型
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost未安装，无法训练XGBoost模型")
    
    print("开始训练XGBoost模型...")
    
    # 创建XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    print("XGBoost模型训练完成")
    print(f"树的数量: {model.n_estimators}")
    print(f"特征重要性总和: {model.feature_importances_.sum():.6f}")
    
    return model

class SequentialTimeSeriesSplit(BaseCrossValidator):
    """
    顺序时间序列交叉验证器
    将数据按顺序划分为训练集和验证集
    """
    def __init__(self, n_splits=5, train_size=0.8):
        self.n_splits = n_splits
        self.train_size = train_size
    
    def split(self, X, y=None, groups=None):
        """生成训练集和验证集的索引"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # 对于交叉验证，我们只做一次划分
        # 因为时间序列数据通常不做多次重叠的划分
        split_idx = int(n_samples * self.train_size)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1  # 只返回一次划分
# def hyperparameter_tuning(model_type, X_train, y_train, cv=5):
#     """
#     超参数调优
    
#     Args:
#         model_type (str): 模型类型 ('decision_tree', 'random_forest', 'xgboost')
#         X_train (numpy.ndarray): 训练特征矩阵
#         y_train (numpy.ndarray): 训练目标变量
#         cv (int): 交叉验证折数
    
#     Returns:
#         调优后的最佳模型
#     """
#     print(f"开始对{model_type}进行超参数调优...")
    
#     if model_type == 'decision_tree':
#         # param_grid = {
#         #     'max_depth': [None, 10, 20, 30],
#         #     'min_samples_split': [2, 5, 10],
#         #     'min_samples_leaf': [1, 2, 4]
#         # }
#         param_grid = {
#         'max_depth': [5, 8, 10, 12],  # 限制最大深度
#         'min_samples_split': [10, 20, 50],  # 增加分裂所需的最小样本数
#         'min_samples_leaf': [5, 10, 20]  # 增加叶子节点的最小样本数
#     }
#         base_model = DecisionTreeRegressor(random_state=42)
        
#     elif model_type == 'random_forest':
#         # param_grid = {
#         #     'n_estimators': [50, 100, 200],
#         #     'max_depth': [None, 10, 20],
#         #     'min_samples_split': [2, 5],
#         #     'min_samples_leaf': [1, 2]
#         # }
#         param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [10, 20],  # 限制深度
#         'min_samples_split': [5, 10],  # 增加分裂所需的最小样本
#         'min_samples_leaf': [2, 5],  # 增加叶子节点的最小样本
#         'max_features': ['sqrt', 'log2', 0.5]  # 这个有效！
#     }
#         base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
#     elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
#         # param_grid = {
#         #     'n_estimators': [50, 100, 200],
#         #     'max_depth': [3, 6, 9],
#         #     'learning_rate': [0.01, 0.1, 0.2],
#         #     'subsample': [0.8, 0.9, 1.0]
#         # }
#         param_grid = {
#         'n_estimators': [50, 100],  # 2个值
#         'max_depth': [3, 4],  # 2个值
#         'learning_rate': [0.01, 0.05],  # 2个值
#         'colsample_bytree': [0.8],  # 固定为1个值
#         'min_child_weight': [3],  # 固定为1个值
#         'reg_alpha': [0],  # 固定为1个值
#         'reg_lambda': [1, 2]  # 2个值
#     }
#         base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
#     else:
#         print(f"不支持的模型类型或XGBoost不可用: {model_type}")
#         return None
    
#     # 网格搜索
#     grid_search = GridSearchCV(
#         base_model, param_grid, cv=cv, 
#         scoring='neg_mean_squared_error', n_jobs=-1
#     )
    
#     grid_search.fit(X_train, y_train)
    
#     print(f"最佳参数: {grid_search.best_params_}")
#     print(f"最佳交叉验证分数: {-grid_search.best_score_:.6f}")
    
#     return grid_search.best_estimator_
def hyperparameter_tuning(model_type, X_train, y_train, cv=5):
    """
    超参数调优
    
    Args:
        model_type (str): 模型类型 ('decision_tree', 'random_forest', 'xgboost')
        X_train (numpy.ndarray): 训练特征矩阵
        y_train (numpy.ndarray): 训练目标变量
        cv (int): 交叉验证折数（不使用，改为顺序划分）
    
    Returns:
        调优后的最佳模型
    """
    print(f"开始对{model_type}进行超参数调优...")
    print("使用顺序划分：前80%为训练集，后20%为验证集")
    
    if model_type == 'decision_tree':
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = DecisionTreeRegressor(random_state=42)
        
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
    else:
        print(f"不支持的模型类型或XGBoost不可用: {model_type}")
        return None
    
    # 使用顺序交叉验证器
    cv_splitter = SequentialTimeSeriesSplit(n_splits=1, train_size=0.8)
    
    # 网格搜索
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_splitter, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {-grid_search.best_score_:.6f}")
    
    return grid_search.best_estimator_
def evaluate_model(model, X, y, dataset_name="数据集"):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 真实目标变量
        dataset_name (str): 数据集名称
    
    Returns:
        dict: 评估指标字典
    """
    # 预测
    y_pred = model.predict(X)
    
    # 计算评估指标
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"\n=== {dataset_name}评估结果 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    
    return metrics, y_pred

def save_model_and_results(model, train_metrics, test_metrics, feature_columns, 
                          y_train, y_train_pred, y_test, y_test_pred, result_dir, model_name):
    """
    保存模型和结果
    
    Args:
        model: 训练好的模型
        train_metrics (dict): 训练集评估指标
        test_metrics (dict): 测试集评估指标
        feature_columns (list): 特征列名
        y_train, y_train_pred: 训练集真实值和预测值
        y_test, y_test_pred: 测试集真实值和预测值
        result_dir (str): 结果保存目录
        model_name (str): 模型名称
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存模型
    model_path = os.path.join(result_dir, f"{model_name}_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)
    else:
        feature_importance = np.zeros(len(feature_columns))
    
    # 2. 保存评估结果
    results = {
        'model_info': {
            'model_type': model_name,
            'timestamp': timestamp,
            'feature_count': len(feature_columns),
            'feature_importance': {
                feature: float(importance) for feature, importance in zip(feature_columns, feature_importance)
            }
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    # 添加模型特定信息
    if hasattr(model, 'n_estimators'):
        results['model_info']['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        results['model_info']['max_depth'] = model.max_depth
    if hasattr(model, 'learning_rate'):
        results['model_info']['learning_rate'] = model.learning_rate
    
    results_path = os.path.join(result_dir, f"{model_name}_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {results_path}")
    
    # # 3. 保存预测结果
    # predictions = {
    #     'train': {
    #         'true_values': y_train.tolist(),
    #         'predicted_values': y_train_pred.tolist()
    #     },
    #     'test': {
    #         'true_values': y_test.tolist(),
    #         'predicted_values': y_test_pred.tolist()
    #     }
    # }
    
    # predictions_path = os.path.join(result_dir, f"{model_name}_predictions_{timestamp}.json")
    # with open(predictions_path, 'w', encoding='utf-8') as f:
    #     json.dump(predictions, f, ensure_ascii=False, indent=2)
    # print(f"预测结果已保存到: {predictions_path}")
    
    # 4. 保存特征重要性CSV
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_path = os.path.join(result_dir, f"{model_name}_feature_importance_{timestamp}.csv")
    feature_importance_df.to_csv(importance_path, index=False, encoding='utf-8')
    print(f"特征重要性已保存到: {importance_path}")
    
    return {
        'model_path': model_path,
        'results_path': results_path,
        # 'predictions_path': predictions_path,
        'importance_path': importance_path
    }

def plot_results(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                feature_columns, model, result_dir, model_name, title="树模型拟合结果"):
    """
    绘制拟合结果图
    
    Args:
        y_train_true (numpy.ndarray): 训练集真实值
        y_train_pred (numpy.ndarray): 训练集预测值
        y_test_true (numpy.ndarray): 测试集真实值
        y_test_pred (numpy.ndarray): 测试集预测值
        feature_columns (list): 特征列名
        model: 训练好的模型
        result_dir (str): 结果保存目录
        model_name (str): 模型名称
        title (str): 图表标题
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 训练集和测试集拟合结果对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_train_true, y_train_pred, alpha=0.6, s=20, color='blue', label='训练集')
    min_val = min(y_train_true.min(), y_train_pred.min())
    max_val = max(y_train_true.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title('训练集拟合结果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test_true, y_test_pred, alpha=0.6, s=20, color='green', label='测试集')
    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('预测值')
    ax2.set_title('测试集拟合结果')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax3 = plt.subplot(2, 3, 3)
    train_residuals = y_train_true - y_train_pred
    test_residuals = y_test_true - y_test_pred
    ax3.scatter(y_train_pred, train_residuals, alpha=0.6, s=20, color='blue', label='训练集残差')
    ax3.scatter(y_test_pred, test_residuals, alpha=0.6, s=20, color='green', label='测试集残差')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('预测值')
    ax3.set_ylabel('残差')
    ax3.set_title('残差图')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. 特征重要性
    ax4 = plt.subplot(2, 3, 4)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_)
    else:
        feature_importance = np.zeros(len(feature_columns))
    
    top_features = min(10, len(feature_columns))
    top_indices = np.argsort(feature_importance)[-top_features:]
    
    ax4.barh(range(top_features), feature_importance[top_indices])
    ax4.set_yticks(range(top_features))
    ax4.set_yticklabels([feature_columns[i] for i in top_indices])
    ax4.set_xlabel('特征重要性')
    ax4.set_title(f'前{top_features}个重要特征')
    ax4.grid(True, alpha=0.3)
    
    # 4. 预测值分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(y_train_true, bins=30, alpha=0.7, label='训练集真实值', color='blue')
    ax5.hist(y_train_pred, bins=30, alpha=0.7, label='训练集预测值', color='red')
    ax5.set_xlabel('值')
    ax5.set_ylabel('频次')
    ax5.set_title('训练集值分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. 测试集预测值分布
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_test_true, bins=30, alpha=0.7, label='测试集真实值', color='green')
    ax6.hist(y_test_pred, bins=30, alpha=0.7, label='测试集预测值', color='orange')
    ax6.set_xlabel('值')
    ax6.set_ylabel('频次')
    ax6.set_title('测试集值分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f"{title} - {model_name}", fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(result_dir, f"{model_name}_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"分析图已保存为: {plot_path}")
    
    plt.show()
    
    return plot_path

def train_and_evaluate_model(model_type, X_train, X_test, y_train, y_test, feature_columns, result_dir, use_tuning=True):
    """
    训练和评估指定类型的模型
    
    Args:
        model_type (str): 模型类型 ('decision_tree', 'random_forest', 'xgboost')
        X_train, X_test, y_train, y_test: 训练和测试数据
        feature_columns (list): 特征列名
        result_dir (str): 结果保存目录
        use_tuning (bool): 是否使用超参数调优
    
    Returns:
        dict: 保存的文件路径
    """
    print(f"\n{'='*50}")
    print(f"开始训练 {model_type} 模型")
    print(f"{'='*50}")
    
    # 训练模型
    if use_tuning:
        model = hyperparameter_tuning(model_type, X_train, y_train)
        if model is None:
            print(f"跳过 {model_type} 模型（不支持或不可用）")
            return None
    else:
        if model_type == 'decision_tree':
            model = train_decision_tree(X_train, y_train)
        elif model_type == 'random_forest':
            model = train_random_forest(X_train, y_train)
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                print(f"跳过 {model_type} 模型（XGBoost不可用）")
                return None
            model = train_xgboost(X_train, y_train)
        else:
            print(f"不支持的模型类型: {model_type}")
            return None
    
    # 评估模型
    train_metrics, y_train_pred = evaluate_model(model, X_train, y_train, f"{model_type} 训练集")
    test_metrics, y_test_pred = evaluate_model(model, X_test, y_test, f"{model_type} 测试集")
    
    # 打印特征重要性
    if hasattr(model, 'feature_importances_'):
        print(f"\n=== {model_type} 特征重要性（按重要性排序）===")
        feature_importance = list(zip(feature_columns, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.6f}")
    
    # 保存模型和结果
    saved_paths = save_model_and_results(
        model, train_metrics, test_metrics, feature_columns,
        y_train, y_train_pred, y_test, y_test_pred, result_dir, model_type
    )
    
    # 绘制结果图
    plot_path = plot_results(
        y_train, y_train_pred, y_test, y_test_pred,
        feature_columns, model, result_dir, model_type
    )
    
    saved_paths['plot_path'] = plot_path
    return saved_paths

def main():
    # 确保结果目录存在
    result_dir = ensure_result_dir()
    
    # CSV文件路径
    csv_file = "./data/束流.csv"
    
    try:
        # 加载和准备数据（使用所有数据）
        X, y, feature_columns = load_and_prepare_data(csv_file, n_samples=None)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # 定义要训练的模型类型
        models_to_train = ['decision_tree', 'random_forest']
        if XGBOOST_AVAILABLE:
            models_to_train.append('xgboost')
        
        # 训练和评估所有模型
        all_results = {}
        for model_type in models_to_train:
            results = train_and_evaluate_model(
                model_type, X_train, X_test, y_train, y_test, 
                feature_columns, result_dir, use_tuning=True
            )
            if results:
                all_results[model_type] = results
        
        # 打印总结
        print(f"\n{'='*60}")
        print("所有模型训练完成！")
        print(f"{'='*60}")
        print("保存的文件:")
        for model_type, paths in all_results.items():
            print(f"\n{model_type} 模型:")
            for key, path in paths.items():
                print(f"  {key}: {os.path.basename(path)}")
        
        print(f"\n所有结果已保存到 {result_dir}")
        print("树模型分析完成！")
        
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
