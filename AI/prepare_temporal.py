import os
import numpy as np
import pandas as pd

WINDOW = 5
NUM_BANDS = 16

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

def make_temporal_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    rows = []

    for source_file, group in df.groupby("source_file"):
        group = group.reset_index(drop=True)

        X = group[band_cols].values.astype(np.float32)
        label = int(group["label"].iloc[0])

        if "sound_type" in group.columns:
            sound_type = group["sound_type"].iloc[0]
        else:
            sound_type = "unknown"

        for start in range(0, len(group) - WINDOW + 1, WINDOW):
            window = X[start:start + WINDOW]

            mean_feat = np.mean(window, axis=0)
            std_feat = np.std(window, axis=0)

            row = {
                "source_file": source_file,
                "sound_type": sound_type,
                "label": label
            }

            for i in range(NUM_BANDS):
                row[f"mean_{i}"] = mean_feat[i]

            for i in range(NUM_BANDS):
                row[f"std_{i}"] = std_feat[i]

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print(f"{input_csv} -> {output_csv}")
    print("Windows:", len(out))

    return out


print("============================================")
print("TEMPORAL FEATURE PREPARATION")
print("WINDOW =", WINDOW)
print("============================================")

train_temporal = make_temporal_dataset(
    "train.csv",
    "train_temporal.csv"
)

val_temporal = make_temporal_dataset(
    "val.csv",
    "val_temporal.csv"
)

test_temporal = make_temporal_dataset(
    "test.csv",
    "test_temporal.csv"
)

print()
print("============================================")
print("SUMMARY")
print("============================================")

print("Train windows:", len(train_temporal))
print("Val windows  :", len(val_temporal))
print("Test windows :", len(test_temporal))

print()
print("Train label distribution:")
print(train_temporal["label"].value_counts().sort_index())

print()
print("Val label distribution:")
print(val_temporal["label"].value_counts().sort_index())

print()
print("Test label distribution:")
print(test_temporal["label"].value_counts().sort_index())