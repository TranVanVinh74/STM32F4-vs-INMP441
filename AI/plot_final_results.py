import glob
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

WINDOW = 5
NUM_BANDS = 16

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

model = keras.models.load_model("mlp_audio_temporal.keras")
scaler = joblib.load("scaler_temporal.pkl")

files = glob.glob("final_test_raw/**/*.csv", recursive=True)

results = []

all_true = []
all_pred = []

# ============================================================
# PREDICT FINAL TEST
# ============================================================

for f in sorted(files):
    df = pd.read_csv(f)

    X_raw = df[band_cols].values.astype(np.float32)

    label = int(df["label"].iloc[0])
    sound_type = df["sound_type"].iloc[0]

    features = []

    for start in range(0, len(X_raw) - WINDOW + 1, WINDOW):
        window = X_raw[start:start + WINDOW]

        mean_feat = np.mean(window, axis=0)
        std_feat = np.std(window, axis=0)

        feat = np.concatenate([mean_feat, std_feat])

        features.append(feat)

    if len(features) == 0:
        continue

    X = np.array(features, dtype=np.float32)

    # Giống hệt preprocessing lúc train
    X = np.log1p(X)
    X = scaler.transform(X)

    probs = model.predict(X, verbose=0).flatten()

    preds = (probs >= 0.5).astype(int)

    accuracy = np.mean(preds == label) * 100

    results.append({
        "sound_type": sound_type,
        "accuracy": accuracy,
        "windows": len(preds)
    })

    all_true.extend([label] * len(preds))
    all_pred.extend(preds)


# ============================================================
# PRINT RESULTS
# ============================================================

print("============================================")
print("FINAL TEST RESULTS")
print("============================================")

for r in results:
    print(
        f"{r['sound_type']:10s} "
        f"accuracy={r['accuracy']:6.2f}% "
        f"windows={r['windows']}"
    )


# ============================================================
# 1. BAR CHART - ACCURACY BY SOUND TYPE
# ============================================================

sound_types = [r["sound_type"] for r in results]
accuracies = [r["accuracy"] for r in results]

plt.figure(figsize=(8, 5))

bars = plt.bar(sound_types, accuracies)

plt.xlabel("Sound Type")
plt.ylabel("Accuracy (%)")
plt.title("Final Blind Test Accuracy by Sound Type")

plt.ylim(0, 100)
plt.grid(axis="y")

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("final_accuracy_by_type.png", dpi=300)
plt.show()


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

all_true = np.array(all_true)
all_pred = np.array(all_pred)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)

print()
print("============================================")
print("CONFUSION MATRIX")
print("============================================")

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NOISE", "KEEP"]
)

disp.plot(values_format="d")

plt.title("Final Blind Test Confusion Matrix")
plt.tight_layout()

plt.savefig("final_confusion_matrix.png", dpi=300)
plt.show()


print()
print("Saved:")
print("final_accuracy_by_type.png")
print("final_confusion_matrix.png")