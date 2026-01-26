import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ======================
# 0. 基本设置
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

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
baseline_df = df.iloc[:10000]          # 正常工况
anomaly_df  = df.iloc[22000:23000]     # 异常区间

X_train = baseline_df[feature_cols].values
X_anom  = anomaly_df[feature_cols].values

# ======================
# 3. 标准化（只用正常数据）
# ======================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_anom_std  = scaler.transform(X_anom)

X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
X_anom_tensor  = torch.tensor(X_anom_std, dtype=torch.float32)

# ======================
# 4. DataLoader
# ======================
dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

input_dim = X_train_tensor.shape[1]

# ======================
# 5. 定义 AutoEncoder
# ======================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = AutoEncoder(input_dim).to(DEVICE)

# ======================
# 6. 训练设置
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
PATIENCE = 10

best_loss = np.inf
patience_cnt = 0

# ======================
# 7. 训练（仅正常工况）
# ======================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for (x_batch,) in train_loader:
        x_batch = x_batch.to(DEVICE)

        optimizer.zero_grad()
        recon = model(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.size(0)

    epoch_loss /= len(train_loader.dataset)

    print(f"Epoch [{epoch+1:03d}/{EPOCHS}] - Train MSE: {epoch_loss:.6f}")

    # Early stopping（用训练误差即可，够用）
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_state = model.state_dict()
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

# 恢复最优模型
model.load_state_dict(best_state)
model.eval()

# ======================
# 8. 异常区间重构
# ======================
with torch.no_grad():
    X_anom_tensor = X_anom_tensor.to(DEVICE)
    X_recon = model(X_anom_tensor).cpu().numpy()

# ======================
# 9. 变量级重构误差
# ======================
recon_error = np.mean(
    (X_anom_std - X_recon) ** 2,
    axis=0
)

ae_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_reconstruction_error": recon_error
}).sort_values("mean_reconstruction_error", ascending=False)

print(ae_df.head(10))

# ======================
# 10. 保存结果
# ======================
ae_df.to_csv(
    "result/anomaly_diagnose/ae_feature_reconstruction_error_torch.csv",
    index=False
)
