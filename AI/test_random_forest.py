import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dùng chính preprocessing cũ để so công bằng với MLP
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy").astype(int)
y_val = np.load("y_val.npy").astype(int)
y_test = np.load("y_test.npy").astype(int)

val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print()
print("============================================")
print("RANDOM FOREST RESULT")
print("============================================")
print(f"Validation accuracy: {accuracy_score(y_val, val_pred) * 100:.2f}%")
print(f"Test accuracy      : {accuracy_score(y_test, test_pred) * 100:.2f}%")


def evaluate_by_file(df, pred, name):
    df = df.copy()
    df["prediction"] = pred
    df["correct"] = df["label"] == df["prediction"]

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for source_file, group in df.groupby("source_file"):
        filename = str(source_file).replace("\\", "/").split("/")[-1]
        acc = group["correct"].mean() * 100

        print(
            f"{filename:18s} "
            f"accuracy={acc:6.2f}%"
        )


evaluate_by_file(val_df, val_pred, "VALIDATION BY FILE")
evaluate_by_file(test_df, test_pred, "TEST BY FILE")