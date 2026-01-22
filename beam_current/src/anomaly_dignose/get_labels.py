import pandas as pd
import numpy as np

# ======================
# 1. 读取与基础设置
# ======================
df = pd.read_csv("data/raw/束流.csv")
df = df.sort_values("时间").reset_index(drop=True)

target_col = "target"

# ======================
# 2. 稳态基准段（前10000点）
# ======================
N_BASELINE = 10000
baseline = df.loc[:N_BASELINE-1, target_col]

mu0 = baseline.mean()
sigma0 = baseline.std()

print(f"Baseline mean: {mu0:.6f}, std: {sigma0:.6f}")

# ======================
# 3. 状态偏离量
# ======================
df["state_deviation"] = (df[target_col] - mu0).abs()

# ======================
# 4. 初始异常状态（幅度判据）
# ======================
K = 3.0  # 3σ 工程阈值
df["is_abnormal_raw"] = df["state_deviation"] > K * sigma0

# ======================
# 5. 持续性约束（关键）
# ======================
WINDOW = 60          # 连续 50 秒
RATIO = 0.1       # 至少 80% 时间异常

df["abnormal_ratio"] = (
    df["is_abnormal_raw"]
    .rolling(WINDOW, min_periods=WINDOW)
    .mean()
)

df["is_abnormal_state"] = df["abnormal_ratio"] > RATIO

df["is_abnormal_state"] = df["is_abnormal_state"].fillna(False)

print(df["is_abnormal_state"].value_counts())

# ======================
# 6. 异常区间提取
# ======================
def extract_intervals(flags, min_len):
    intervals = []
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                intervals.append((start, i - 1))
            start = None

    if start is not None and len(flags) - start >= min_len:
        intervals.append((start, len(flags) - 1))

    return intervals


anomaly_intervals = extract_intervals(
    df["is_abnormal_state"].values,
    min_len=WINDOW
)

print("Final anomaly intervals:")
for s, e in anomaly_intervals:
    print(f"[{s}, {e}]")




import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))

# 原始束流
plt.plot(df[target_col], label="Beam Intensity", linewidth=1.2)

# 用红色背景标注异常区间
for i, (start, end) in enumerate(anomaly_intervals):
    plt.axvspan(
        start,
        end,
        color="red",
        alpha=0.25,
        label="Anomaly Interval" if i == 0 else None
    )

plt.xlabel("Time Index")
plt.ylabel("Beam Intensity")
plt.title("Beam Intensity with Anomaly Intervals (3σ Rule)")
plt.legend()
plt.tight_layout()
plt.show()

print(df.columns)
# 2. 先将is_abnormal_raw重命名为is_abnormal（避免重命名前被删除）
df.rename(columns={"is_abnormal_raw": "is_abnormal"}, inplace=True)
# 把is_abnormal的True和False换成1和0
df["is_abnormal"] = df["is_abnormal"].astype(int)

# 3. 定义需要删除的列（注意：这里已经去掉了is_abnormal_raw，因为已重命名）
columns_to_drop = ["state_deviation", "abnormal_ratio", "is_abnormal_state"]

# 4. 执行删除操作（errors='ignore'避免列不存在时报错）
df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

# 5. 保存处理后的数据（可选）
df.to_csv("data/raw/束流_labels.csv", index=False)

# import matplotlib.pyplot as plt

# # ========= 可视化：原始束流 + 异常点标红 =========
# plt.figure(figsize=(14, 4))

# # 原始束流时间序列
# plt.plot(df[target_col], label="Beam Intensity")

# # 异常点索引
# anomaly_idx = df.index[df["is_anomaly_point"]]

# # 异常点标红
# plt.scatter(
#     anomaly_idx,
#     df.loc[anomaly_idx, target_col],
#     s=12,
#     label="Anomaly Points",
#     c="red"
# )

# plt.xlabel("Time Index")
# plt.ylabel("Beam Intensity")
# plt.title("Beam Intensity with Anomaly Points (3σ on Differenced Signal)")
# plt.legend()
# plt.tight_layout()
# plt.show()