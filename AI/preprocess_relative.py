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


def relative_features(X):
    # Nén dynamic range
    X = np.log1p(X)

    # Loại bớt ảnh hưởng âm lượng tổng thể của từng frame
    frame_mean = np.mean(X, axis=1, keepdims=True)
    X = X - frame_mean

    return X


X_train = relative_features(X_train)
X_val = relative_features(X_val)
X_test = relative_features(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("============================================")
print("RELATIVE PREPROCESS CHECK")
print("============================================")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print()
print("Train global mean:", X_train.mean())
print("Train global std :", X_train.std())

np.save("X_train_relative.npy", X_train)
np.save("X_val_relative.npy", X_val)
np.save("X_test_relative.npy", X_test)

np.save("y_train_relative.npy", y_train)
np.save("y_val_relative.npy", y_val)
np.save("y_test_relative.npy", y_test)

joblib.dump(scaler, "scaler_relative.pkl")

print()
print("Saved relative preprocessing files.")