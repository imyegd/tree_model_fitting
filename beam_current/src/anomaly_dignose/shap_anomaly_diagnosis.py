# import numpy as np
# import pandas as pd
# import joblib
# import shap

# from sklearn.model_selection import train_test_split

# # ===============================
# # 1. 基本配置
# # ===============================

# DATA_PATH = "data/raw/束流.csv"
# ARTIFACT_DIR = "models"
# N_BASELINE = 10000
# TOP_K = 5
# RANDOM_STATE = 42

# target_col = "target"

# # ===============================
# # 2. 读取数据 & 基本处理
# # ===============================

# df = pd.read_csv(DATA_PATH)
# df = df.sort_values("时间").reset_index(drop=True)

# feature_cols = [c for c in df.columns if c not in ["时间", target_col]]

# X = df[feature_cols].values
# y = df[target_col].values

# # ===============================
# # 3. 加载“正常范围”参数
# # ===============================

# normal_stats = np.load(
#     f"{ARTIFACT_DIR}/normal_stats.npy",
#     allow_pickle=True
# ).item()

# mu_normal = normal_stats["mu"]
# sigma_normal = normal_stats["sigma"]

# print(f"[INFO] Normal mean={mu_normal:.4f}, std={sigma_normal:.4f}")

# # ===============================
# # 4. 重新划分训练 / 测试集（保持一致）
# # ===============================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     shuffle=True,
#     random_state=RANDOM_STATE
# )

# X_test_df = pd.DataFrame(X_test, columns=feature_cols)
# y_test_series = pd.Series(y_test)

# # ===============================
# # 5. 确定“测试集异常样本”
# # （基于工程定义，而非模型预测）
# # ===============================

# test_anomaly_mask = (
#     (y_test_series - mu_normal).abs() > 3 * sigma_normal
# )

# X_test_anomaly = X_test_df.loc[test_anomaly_mask.values]

# print(f"[INFO] Anomaly samples in test set: {X_test_anomaly.shape[0]}")

# if X_test_anomaly.shape[0] == 0:
#     raise RuntimeError("测试集中未检测到异常样本，无法进行 SHAP 诊断")

# # ===============================
# # 6. 加载模型
# # ===============================

# models = {
#     "Linear": joblib.load(f"{ARTIFACT_DIR}/Linear_regressor.pkl"),
#     "RF": joblib.load(f"{ARTIFACT_DIR}/RF_regressor.pkl"),
#     "XGB": joblib.load(f"{ARTIFACT_DIR}/XGB_regressor.pkl"),
#     "LGBM": joblib.load(f"{ARTIFACT_DIR}/LGBM_regressor.pkl"),
# }

# # ===============================
# # 7. SHAP 计算
# # ===============================

# def compute_shap(model_name, model, X_train, X_anomaly):
#     if model_name == "Linear":
#         # Pipeline: scaler + linear
#         explainer = shap.Explainer(
#             model.named_steps["model"],
#             X_train
#         )
#         shap_values = explainer(X_anomaly.values).values

#     elif model_name in ["RF", "XGB", "LGBM"]:
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(X_anomaly)

#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

#     return shap_values


# shap_results = {}

# for name, model in models.items():
#     print(f"[INFO] Computing SHAP for {name}...")
#     shap_vals = compute_shap(
#         name,
#         model,
#         X_train,
#         X_test_anomaly
#     )
#     shap_results[name] = shap_vals

# # ===============================
# # 8. 提取 Top-K 关键变量
# # ===============================

# def get_top_features(shap_values, feature_names, top_k=5):
#     mean_abs_shap = np.abs(shap_values).mean(axis=0)
#     df = pd.DataFrame({
#         "feature": feature_names,
#         "mean_abs_shap": mean_abs_shap
#     })
#     df = df.sort_values("mean_abs_shap", ascending=False)
#     return df.head(top_k)


# top_features = {}

# for name, shap_vals in shap_results.items():
#     top_df = get_top_features(
#         shap_vals,
#         feature_cols,
#         TOP_K
#     )
#     top_features[name] = top_df
#     print(f"\n[{name}] Top-{TOP_K} features:")
#     print(top_df)

# # ===============================
# # 9. 关键变量一致性分析
# # ===============================

# def consistency_score(feature_lists):
#     sets = [set(lst) for lst in feature_lists]
#     intersection = set.intersection(*sets)
#     union = set.union(*sets)
#     score = len(intersection) / len(union) if len(union) > 0 else 0.0
#     return intersection, score


# feature_lists = [
#     top_features[name]["feature"].tolist()
#     for name in top_features
# ]

# common_features, consistency = consistency_score(feature_lists)

# print("\n[CONSISTENCY RESULT]")
# print("Common features:", common_features)
# print(f"Consistency score: {consistency:.4f}")

# # ===============================
# # 10. 保存结果（方便论文和复现）
# # ===============================

# for name, df_top in top_features.items():
#     df_top.to_csv(
#         f"{ARTIFACT_DIR}/shap_top_features_{name}.csv",
#         index=False
#     )

# pd.DataFrame({
#     "common_features": list(common_features),
#     "consistency_score": [consistency]
# }).to_csv(
#     f"{ARTIFACT_DIR}/shap_consistency.csv",
#     index=False
# )

# print("\n[INFO] SHAP diagnosis finished.")

import pandas as pd
import numpy as np

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("data/raw/束流.csv")
df = df.sort_values("时间").reset_index(drop=True)

target_col = "target"
feature_cols = [c for c in df.columns if c not in ["时间", target_col]]

# ======================
# 2. 选取数据区间
# ======================
baseline_df = df.iloc[:10000]
anomaly_df = df.iloc[22000:23000]

# ======================
# 3. 统计差异分析
# ======================
results = []

for col in feature_cols:
    mu_normal = baseline_df[col].mean()
    mu_anomaly = anomaly_df[col].mean()
    std_normal = baseline_df[col].std()

    z_score = (mu_anomaly - mu_normal) / (std_normal + 1e-6)

    results.append({
        "feature": col,
        "mean_normal": mu_normal,
        "mean_anomaly": mu_anomaly,
        "z_score": z_score,
        "abs_z": abs(z_score)
    })

result_df = pd.DataFrame(results)
result_df = result_df.sort_values("abs_z", ascending=False)

# ======================
# 4. 输出 Top-k
# ======================
print(result_df.head(10))

result_df.to_csv("result/anomaly_diagnose/stat_diff_results.csv", index=False)


import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("data/raw/束流.csv")
df = df.sort_values("时间").reset_index(drop=True)

target_col = "target"
feature_cols = [c for c in df.columns if c not in ["时间", target_col]]

# ======================
# 2. 数据区间
# ======================
baseline_df = df.iloc[:10000]
anomaly_df = df.iloc[22000:23000]

X_train = baseline_df[feature_cols]
y_train = baseline_df[target_col]

X_anomaly = anomaly_df[feature_cols]

# ======================
# 3. 标准化
# ======================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_anomaly_std = scaler.transform(X_anomaly)

# ======================
# 4. PLS 建模
# ======================
pls = PLSRegression(n_components=5)
pls.fit(X_train_std, y_train)

# ======================
# 5. 特征贡献（权重）
# ======================
weights = np.abs(pls.x_weights_[:, 0])  # 第一主成分
weights = weights / weights.sum()

pls_df = pd.DataFrame({
    "feature": feature_cols,
    "pls_weight": weights
}).sort_values("pls_weight", ascending=False)

print(pls_df.head(10))

pls_df.to_csv("result/anomaly_diagnose/pls_feature_contribution.csv", index=False)



import pandas as pd
import numpy as np
import shap
import joblib

# ======================
# 1. 读取数据 & 模型
# ======================
df = pd.read_csv("data/raw/束流.csv")
df = df.sort_values("时间").reset_index(drop=True)

model = joblib.load("models\RF_regressor.pkl")  # 你自己的路径

target_col = "target"
feature_cols = [c for c in df.columns if c not in ["时间", target_col]]

# ======================
# 2. 选取异常区间
# ======================
anomaly_df = df.iloc[22000:23000]
X_anomaly = anomaly_df[feature_cols]

# ======================
# 3. SHAP 计算
# ======================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_anomaly)

# ======================
# 4. 计算平均 |SHAP|
# ======================
shap_mean = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": shap_mean
}).sort_values("mean_abs_shap", ascending=False)

print(shap_df.head(10))

shap_df.to_csv("result/anomaly_diagnose/shap_rf_anomaly.csv", index=False)

# ======================
# 5. 可选：画图
# ======================
shap.summary_plot(shap_values, X_anomaly, show=True)



# 比如 Top-10 overlap
set_stat = set(result_df.head(10)["feature"])
set_shap = set(shap_df.head(10)["feature"])
set_pls  = set(pls_df.head(10)["feature"])

overlap = set_stat & set_shap & set_pls
print(overlap)





