import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib

df = pd.read_csv("data/raw/束流.csv")
df = df.sort_values("时间").reset_index(drop=True)

target_col = "target"
feature_cols = [c for c in df.columns if c not in ["时间", target_col]]

N_BASELINE = 10000

baseline_target = df.loc[:N_BASELINE-1, target_col]

mu_normal = baseline_target.mean()
sigma_normal = baseline_target.std()

print(f"Normal mean: {mu_normal:.4f}, std: {sigma_normal:.4f}")

# 保存正常范围参数
np.save("data/processed/normal_stats.npy", {
    "mu": mu_normal,
    "sigma": sigma_normal
})

X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)


models = {}

models["Linear"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

models["RF"] = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

from xgboost import XGBRegressor

models["XGB"] = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

from lightgbm import LGBMRegressor

models["LGBM"] = LGBMRegressor(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

models["MLP"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=500,
        random_state=42
    ))
])

reg_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    reg_results.append({
        "model": name,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    joblib.dump(model, f"artifacts/{name}_regressor.pkl")

reg_results = pd.DataFrame(reg_results)
print(reg_results)

