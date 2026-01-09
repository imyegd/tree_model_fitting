import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 假设您的 X_train, X_test, y_train, y_test 变量已加载到环境中。
# 如果需要重新加载，请取消注释以下代码：
X_train = pd.read_csv('./data/processed/X_train.csv')
X_test = pd.read_csv('./data/processed/X_test.csv')
y_train = pd.read_csv('./data/processed/y_train.csv')
y_test = pd.read_csv('./data/processed/y_test.csv')

print("--- 开始训练 XGBoost Regressor ---")

# 1. 初始化基模型
# 使用一些默认的、性能较好的参数
xgb_base_model = XGBRegressor(
    n_estimators=500,       # 树的数量
    max_depth=7,            # 树的最大深度，控制复杂度
    learning_rate=0.05,     # 学习率，控制收敛速度
    subsample=0.7,          # 采样比例，防止过拟合
    colsample_bytree=0.7,   # 列采样比例
    random_state=42,
    n_jobs=-1               # 使用所有核心加速训练
)

# 2. 包装为多输出模型
# MultiOutputRegressor 会为 Delta_X 和 Delta_Y 各自训练一个独立的 XGBoost 模型
xgb_model = MultiOutputRegressor(xgb_base_model)

# 3. 训练模型
# XGBoost 不需要标准化，使用原始数据即可
xgb_model.fit(X_train, y_train)

# 4. 预测
y_pred_xgb = xgb_model.predict(X_test)

# 5. 评估
# 使用加权平均 R2 分数评估多输出模型
r2_xgb = r2_score(y_test, y_pred_xgb, multioutput='variance_weighted')
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print("\n--- 🚀 XGBoost 性能评估 (测试集) 🚀 ---")
print(f"XGBoost R-squared (加权平均): {r2_xgb:.4f}")
print(f"XGBoost Mean Squared Error: {mse_xgb:.4f}")

# 6. 分别查看 Delta_X 和 Delta_Y 的 R2
# 注意：y_test 是一个 DataFrame，y_pred_xgb 是一个 NumPy 数组
r2_x = r2_score(y_test['Delta_X'], y_pred_xgb[:, 0])
r2_y = r2_score(y_test['Delta_Y'], y_pred_xgb[:, 1])

print(f"Delta_X (束位X差值) R-squared: {r2_x:.4f}")
print(f"Delta_Y (束位Y差值) R-squared: {r2_y:.4f}")