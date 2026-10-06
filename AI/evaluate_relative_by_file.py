import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model("mlp_audio_relative.keras")

def evaluate_split(csv_file, npy_file, name):
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)

    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    df["prediction"] = preds
    df["prob_keep"] = probs
    df["correct"] = df["label"] == df["prediction"]

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for source_file, group in df.groupby("source_file"):
        filename = str(source_file).replace("\\", "/").split("/")[-1]
        acc = group["correct"].mean() * 100
        avg_prob = group["prob_keep"].mean()

        print(
            f"{filename:18s} "
            f"accuracy={acc:6.2f}%  "
            f"avg_keep_prob={avg_prob:.4f}"
        )

evaluate_split("val.csv", "X_val_relative.npy", "VALIDATION")
evaluate_split("test.csv", "X_test_relative.npy", "TEST")