import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

NUM_BANDS = 16

mean_cols = [f"mean_{i}" for i in range(NUM_BANDS)]
std_cols = [f"std_{i}" for i in range(NUM_BANDS)]
feature_cols = mean_cols + std_cols

train_df = pd.read_csv("train_temporal_10.csv")
val_df = pd.read_csv("val_temporal_10.csv")
test_df = pd.read_csv("test_temporal_10.csv")

X_train = train_df[feature_cols].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

y_train = train_df["label"].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.float32)

X_train = np.log1p(X_train)
X_val = np.log1p(X_val)
X_test = np.log1p(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("============================================")
print("TEMPORAL 10 PREPROCESS CHECK")
print("============================================")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print()
print("Train mean:", X_train.mean())
print("Train std :", X_train.std())

np.save("X_train_temporal_10.npy", X_train)
np.save("X_val_temporal_10.npy", X_val)
np.save("X_test_temporal_10.npy", X_test)

np.save("y_train_temporal_10.npy", y_train)
np.save("y_val_temporal_10.npy", y_val)
np.save("y_test_temporal_10.npy", y_test)

joblib.dump(scaler, "scaler_temporal_10.pkl")

print()
print("Saved temporal 10 preprocessed files.")