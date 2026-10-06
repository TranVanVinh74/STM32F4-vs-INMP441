import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

WINDOW = 5
NUM_BANDS = 16
NUM_FEATURES = WINDOW * NUM_BANDS

feature_cols = [f"f_{i}" for i in range(NUM_FEATURES)]

train_df = pd.read_csv("train_concat.csv")
val_df = pd.read_csv("val_concat.csv")
test_df = pd.read_csv("test_concat.csv")

X_train = train_df[feature_cols].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

y_train = train_df["label"].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.float32)

# Nén dynamic range của power
X_train = np.log1p(X_train)
X_val = np.log1p(X_val)
X_test = np.log1p(X_test)

# Scale theo thống kê của train
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("============================================")
print("CONCAT PREPROCESS CHECK")
print("============================================")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print()
print("Train mean:", X_train.mean())
print("Train std :", X_train.std())

np.save("X_train_concat.npy", X_train)
np.save("X_val_concat.npy", X_val)
np.save("X_test_concat.npy", X_test)

np.save("y_train_concat.npy", y_train)
np.save("y_val_concat.npy", y_val)
np.save("y_test_concat.npy", y_test)

joblib.dump(scaler, "scaler_concat.pkl")

print()
print("Saved concat preprocessed files.")