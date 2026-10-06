import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

NUM_BANDS = 16

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df[band_cols].values.astype(np.float32)
X_val = val_df[band_cols].values.astype(np.float32)
X_test = test_df[band_cols].values.astype(np.float32)

y_train = train_df["label"].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.float32)

# Log compression
X_train_log = np.log1p(X_train)
X_val_log = np.log1p(X_val)
X_test_log = np.log1p(X_test)

# Normalization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)
X_test_scaled = scaler.transform(X_test_log)

print("============================================")
print("PREPROCESS CHECK")
print("============================================")

print("X_train:", X_train_scaled.shape)
print("X_val  :", X_val_scaled.shape)
print("X_test :", X_test_scaled.shape)

print()
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

print()
print("Train feature mean:")
print(np.round(X_train_scaled.mean(axis=0), 4))

print()
print("Train feature std:")
print(np.round(X_train_scaled.std(axis=0), 4))

print()
print("Scaler mean:")
print(scaler.mean_)

print()
print("Scaler scale:")
print(scaler.scale_)

np.save("X_train.npy", X_train_scaled)
np.save("X_val.npy", X_val_scaled)
np.save("X_test.npy", X_test_scaled)

np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)

joblib.dump(scaler, "scaler.pkl")

print()
print("Saved:")
print("X_train.npy")
print("X_val.npy")
print("X_test.npy")
print("y_train.npy")
print("y_val.npy")
print("y_test.npy")
print("scaler.pkl")