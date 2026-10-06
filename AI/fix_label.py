import pandas as pd

files = [
    "dataset_raw/noise/keyboard/keyboard_04.csv",
    "dataset_raw/noise/keyboard/keyboard_05.csv",
]

for f in files:
    df = pd.read_csv(f)

    print(f"{f}: old labels =", df["label"].unique())

    df["label"] = 0

    df.to_csv(f, index=False)

    print(f"{f}: fixed -> label=0")