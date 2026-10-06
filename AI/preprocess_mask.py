import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

NUM_BANDS = 16
band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

X_train_raw = train_df[band_cols].values.astype(np.float32)
X_val_raw = val_df[band_cols].values.astype(np.float32)
X_test_raw = test_df[band_cols].values.astype(np.float32)

y_train = train_df["label"].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.float32)

# 16 binary features: band có dữ liệu hay không
M_train = (X_train_raw > 0).astype(np.float32)
M_val = (X_val_raw > 0).astype(np.float32)
M_test = (X_test_raw > 0).astype(np.float32)

# 16 power features
X_train_log = np.log1p(X_train_raw)
X_val_log = np.log1p(X_val_raw)
X_test_log = np.log1p(X_test_raw)

scaler = StandardScaler()

X_train_power = scaler.fit_transform(X_train_log)
X_val_power = scaler.transform(X_val_log)
X_test_power = scaler.transform(X_test_log)

# Ghép: 16 power + 16 mask = 32 features
X_train = np.concatenate([X_train_power, M_train], axis=1)
X_val = np.concatenate([X_val_power, M_val], axis=1)
X_test = np.concatenate([X_test_power, M_test], axis=1)

print("============================================")
print("POWER + MASK PREPROCESS")
print("============================================")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print()
print("Example mask:")
print(M_train[0])

np.save("X_train_mask.npy", X_train)
np.save("X_val_mask.npy", X_val)
np.save("X_test_mask.npy", X_test)

np.save("y_train_mask.npy", y_train)
np.save("y_val_mask.npy", y_val)
np.save("y_test_mask.npy", y_test)

joblib.dump(scaler, "scaler_mask.pkl")

print()
print("Saved mask dataset.")