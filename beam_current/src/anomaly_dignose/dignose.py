import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("data/raw/束流_labels.csv")
feature_cols = [c for c in df.columns if c not in ["is_abnormal", "时间", "target"]]
X = df[feature_cols].values
y = df["is_abnormal"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

normal = X_train[y_train == 0]
abnormal = X_train[y_train == 1]

stats = []

for i, f in enumerate(feature_cols):
    stats.append({
        "feature": f,
        "delta_mean": abnormal[:, i].mean() - normal[:, i].mean(),
        "delta_std": abnormal[:, i].std() - normal[:, i].std()
    })

stat_df = pd.DataFrame(stats).sort_values(
    by="delta_std", key=np.abs, ascending=False
)

print(stat_df.head(10))

# random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))


rf_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance.head(10)


# import shap

# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_train)

# shap_importance = np.abs(shap_values[1]).mean(axis=0)

# shap_df = pd.DataFrame({
#     "feature": feature_cols,
#     "mean_abs_shap": shap_importance
# }).sort_values("mean_abs_shap", ascending=False)

# shap_df.head(10)

from sklearn.neural_network import MLPRegressor

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(normal)
X_test_scaled = scaler.transform(X_test)

ae = MLPRegressor(
    hidden_layer_sizes=(64, 32, 64),
    max_iter=300,
    random_state=42
)

ae.fit(X_train_scaled, X_train_scaled)

X_recon = ae.predict(X_test_scaled)
recon_error = np.mean((X_test_scaled - X_recon) ** 2, axis=1)

# 用训练集正常误差定阈值
train_recon = ae.predict(X_train_scaled)
train_error = np.mean((X_train_scaled - train_recon) ** 2, axis=1)

threshold = np.percentile(train_error, 95)

y_pred_ae = (recon_error > threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_ae))
print("F1:", f1_score(y_test, y_pred_ae))
error_by_feature = np.mean(
    (X_test_scaled[y_pred_ae == 1] -
     X_recon[y_pred_ae == 1]) ** 2,
    axis=0
)

ae_df = pd.DataFrame({
    "feature": feature_cols,
    "recon_error": error_by_feature
}).sort_values("recon_error", ascending=False)

ae_df.head(10)


from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

X = df[feature_cols].values
y = df["is_abnormal"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pls = PLSRegression(n_components=2)
pls.fit(X_scaled, y)

pls_weights = np.abs(pls.x_weights_).sum(axis=1)

pls_importance = pd.DataFrame({
    "feature": feature_cols,
    "pls_weight": pls_weights
}).sort_values("pls_weight", ascending=False)

pls_importance.head(10)

X_pls = pls.transform(X_scaled)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_tr, X_te, y_tr, y_te = train_test_split(
    X_pls, y, test_size=0.3, stratify=y, random_state=42
)

clf = LogisticRegression()
clf.fit(X_tr, y_tr.ravel())

y_pred = clf.predict(X_te)
print("F1:", f1_score(y_te, y_pred))
print("Accuracy:", accuracy_score(y_te, y_pred))




