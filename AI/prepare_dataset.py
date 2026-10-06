import os
import glob
import pandas as pd

DATASET_DIR = "dataset_raw"
NUM_BANDS = 16

files = glob.glob(os.path.join(DATASET_DIR, "**", "*.csv"), recursive=True)

all_df = []

print("============================================")
print("DATASET FILES")
print("============================================")

for f in files:
    df = pd.read_csv(f)

    required_cols = [f"band_{i}" for i in range(NUM_BANDS)] + ["label"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"SKIP: {f}")
        print("Missing columns:", missing)
        continue

    df["source_file"] = f

    all_df.append(df)

    print(f"{f}: {len(df)} samples, label={df['label'].unique()}")

if not all_df:
    raise RuntimeError("No valid CSV files found.")

data = pd.concat(all_df, ignore_index=True)

print()
print("============================================")
print("MERGED DATASET")
print("============================================")
print("Total samples:", len(data))

print()
print("Label distribution:")
print(data["label"].value_counts().sort_index())

print()
print("Number of files:", data["source_file"].nunique())

print()
print("Samples per sound type:")
if "sound_type" in data.columns:
    print(data.groupby(["sound_type", "label"]).size())
else:
    print("No sound_type column found.")

output_file = "dataset_merged.csv"
data.to_csv(output_file, index=False)

print()
print("Saved:", output_file)