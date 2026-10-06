import pandas as pd
import numpy as np
import glob
import os

band_cols = [f"band_{i}" for i in range(16)]

files = glob.glob("dataset_raw/**/*.csv", recursive=True)

for f in sorted(files):
    df = pd.read_csv(f)

    X = df[band_cols].values.astype(np.float64)

    # Chuẩn hóa từng sample theo tổng năng lượng
    total = X.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0

    X_shape = X / total

    mean_shape = X_shape.mean(axis=0)

    filename = os.path.basename(f)

    print()
    print("=" * 80)
    print(filename)
    print("=" * 80)

    for i, value in enumerate(mean_shape):
        print(f"band_{i:02d}: {value * 100:6.2f}%")