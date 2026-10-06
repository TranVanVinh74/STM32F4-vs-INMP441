import glob
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

WINDOW = 5
NUM_BANDS = 16

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

model = keras.models.load_model("mlp_audio_temporal.keras")
scaler = joblib.load("scaler_temporal.pkl")

files = glob.glob("final_test_raw/**/*.csv", recursive=True)

print("=" * 70)
print("FINAL BLIND TEST")
print("=" * 70)

all_true = []
all_pred = []

for f in sorted(files):
    df = pd.read_csv(f)

    X_raw = df[band_cols].values.astype(np.float32)
    label = int(df["label"].iloc[0])

    features = []

    for start in range(0, len(X_raw) - WINDOW + 1, WINDOW):
        window = X_raw[start:start + WINDOW]

        mean_feat = np.mean(window, axis=0)
        std_feat = np.std(window, axis=0)

        feat = np.concatenate([mean_feat, std_feat])
        features.append(feat)

    if len(features) == 0:
        print(f"{f}: not enough samples")
        continue

    X = np.array(features, dtype=np.float32)

    # giống preprocessing lúc train
    X = np.log1p(X)
    X = scaler.transform(X)

    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    accuracy = np.mean(preds == label) * 100
    avg_prob = np.mean(probs)

    filename = os.path.basename(f)

    print(
        f"{filename:18s} "
        f"label={label} "
        f"windows={len(preds):4d} "
        f"accuracy={accuracy:6.2f}% "
        f"avg_keep_prob={avg_prob:.4f}"
    )

    all_true.extend([label] * len(preds))
    all_pred.extend(preds)

all_true = np.array(all_true)
all_pred = np.array(all_pred)

overall = np.mean(all_true == all_pred) * 100

print()
print("=" * 70)
print(f"OVERALL FINAL ACCURACY: {overall:.2f}%")
print("=" * 70)