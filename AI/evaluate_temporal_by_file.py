import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix

model = keras.models.load_model("mlp_audio_temporal.keras")


def evaluate(csv_file, npy_file, name):
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)

    y = df["label"].values.astype(int)

    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    df["prediction"] = preds
    df["prob_keep"] = probs
    df["correct"] = df["label"] == df["prediction"]

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    cm = confusion_matrix(y, preds)

    print()
    print("CONFUSION MATRIX")
    print("              Pred NOISE   Pred KEEP")
    print(f"Actual NOISE     {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"Actual KEEP      {cm[1,0]:5d}        {cm[1,1]:5d}")

    print()
    print("BY FILE")
    print("-" * 65)

    for source_file, group in df.groupby("source_file"):
        filename = str(source_file).replace("\\", "/").split("/")[-1]

        acc = group["correct"].mean() * 100
        avg_prob = group["prob_keep"].mean()

        print(
            f"{filename:18s} "
            f"accuracy={acc:6.2f}%  "
            f"avg_keep_prob={avg_prob:.4f}"
        )


evaluate(
    "val_temporal.csv",
    "X_val_temporal.npy",
    "VALIDATION TEMPORAL"
)

evaluate(
    "test_temporal.csv",
    "X_test_temporal.npy",
    "TEST TEMPORAL"
)