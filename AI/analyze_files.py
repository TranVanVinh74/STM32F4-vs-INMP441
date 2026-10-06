import glob
import os
import numpy as np
import pandas as pd

files = glob.glob("dataset_raw/**/*.csv", recursive=True)

print("=" * 95)
print(f"{'FILE':18s} {'TYPE':10s} {'LABEL':5s} {'SAMPLES':7s} "
      f"{'MEAN_BIN':9s} {'NZ_BANDS':9s} {'POWER':12s}")
print("=" * 95)

for f in sorted(files):
    df = pd.read_csv(f)

    filename = os.path.basename(f)
    sound_type = df["sound_type"].iloc[0]
    label = int(df["label"].iloc[0])

    bands = df[[f"band_{i}" for i in range(16)]].values

    nonzero = np.count_nonzero(bands > 0, axis=1)
    power = np.sum(bands, axis=1)

    mean_bin = df["bin_count"].mean()
    mean_nonzero = nonzero.mean()
    median_power = np.median(power)

    print(
        f"{filename:18s} "
        f"{sound_type:10s} "
        f"{label:<5d} "
        f"{len(df):<7d} "
        f"{mean_bin:<9.2f} "
        f"{mean_nonzero:<9.2f} "
        f"{median_power:<12.3e}"
    )