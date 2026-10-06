import numpy as np
import pandas as pd

WINDOW = 5
NUM_BANDS = 16

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

def make_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    rows = []

    for source_file, group in df.groupby("source_file", sort=False):
        group = group.reset_index(drop=True)

        X = group[band_cols].values.astype(np.float32)
        label = int(group["label"].iloc[0])
        sound_type = group["sound_type"].iloc[0]

        for start in range(0, len(X) - WINDOW + 1, WINDOW):
            window = X[start:start + WINDOW]

            # 5 x 16 -> 80
            features = window.flatten()

            row = {
                "source_file": source_file,
                "sound_type": sound_type,
                "label": label
            }

            for i, value in enumerate(features):
                row[f"f_{i}"] = value

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print(f"{input_csv} -> {output_csv}")
    print("Windows:", len(out))

    return out


print("============================================")
print("TEMPORAL CONCAT DATASET")
print("============================================")

train = make_dataset("train.csv", "train_concat.csv")
val = make_dataset("val.csv", "val_concat.csv")
test = make_dataset("test.csv", "test_concat.csv")

print()
print("Train:", len(train))
print("Val  :", len(val))
print("Test :", len(test))

print()
print("Feature count:", WINDOW * NUM_BANDS)