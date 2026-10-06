import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# LOAD MODEL + DATA
# =========================

model = keras.models.load_model("mlp_audio.keras")

X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")

y_val = np.load("y_val.npy").astype(int)
y_test = np.load("y_test.npy").astype(int)

val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")


def evaluate(name, X, y, df):
    print()
    print("============================================")
    print(name)
    print("============================================")

    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y, preds)

    print()
    print("CONFUSION MATRIX")
    print("Rows = Actual, Columns = Predicted")
    print()
    print("              Pred NOISE   Pred KEEP")
    print(f"Actual NOISE     {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"Actual KEEP      {cm[1,0]:5d}        {cm[1,1]:5d}")

    print()
    print("CLASSIFICATION REPORT")
    print(classification_report(
        y,
        preds,
        target_names=["NOISE", "KEEP"],
        digits=4
    ))

    # Accuracy từng loại âm thanh
    result = df.copy()
    result["prediction"] = preds
    result["prob_keep"] = probs
    result["correct"] = result["label"] == result["prediction"]

    print()
    print("ACCURACY BY SOUND TYPE")
    print("--------------------------------------------")

    for sound_type, group in result.groupby("sound_type"):
        accuracy = group["correct"].mean() * 100

        print(
            f"{sound_type:12s}: "
            f"{accuracy:6.2f}% "
            f"({group['correct'].sum()}/{len(group)})"
        )

    print()
    print("AVERAGE KEEP PROBABILITY")
    print("--------------------------------------------")

    for sound_type, group in result.groupby("sound_type"):
        print(
            f"{sound_type:12s}: "
            f"{group['prob_keep'].mean():.4f}"
        )


evaluate("VALIDATION", X_val, y_val, val_df)
evaluate("TEST", X_test, y_test, test_df)