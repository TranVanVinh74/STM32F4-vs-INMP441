import pandas as pd

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = r"noise\fan\fan_09.csv"

NUM_BANDS = 16


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

print("============================================")
print("DATASET CHECK")
print("============================================")

print("File:", CSV_FILE)
print("Total samples:", len(df))
print()


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "detected_angle",
    "bin_count"
]

for i in range(NUM_BANDS):
    required_columns.append(f"band_{i}")

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("ERROR: Missing columns:")

    for col in missing_columns:
        print(" -", col)

    raise SystemExit


# ============================================================
# BAND COLUMNS
# ============================================================

band_cols = [
    f"band_{i}"
    for i in range(NUM_BANDS)
]


# ============================================================
# 1. BIN COUNT
# ============================================================

print("============================================")
print("BIN COUNT")
print("============================================")

print("Mean bin_count:", round(df["bin_count"].mean(), 2))
print("Median bin_count:", round(df["bin_count"].median(), 2))
print("Min bin_count:", df["bin_count"].min())
print("Max bin_count:", df["bin_count"].max())

print()

print(
    "Samples bin_count >= 3:",
    round((df["bin_count"] >= 3).mean() * 100, 2),
    "%"
)

print(
    "Samples bin_count >= 5:",
    round((df["bin_count"] >= 5).mean() * 100, 2),
    "%"
)

print(
    "Samples bin_count >= 8:",
    round((df["bin_count"] >= 8).mean() * 100, 2),
    "%"
)

print()


# ============================================================
# 2. NON-ZERO BANDS
# ============================================================

df["nonzero_bands"] = (
    df[band_cols]
    .gt(0)
    .sum(axis=1)
)

print("============================================")
print("NON-ZERO BANDS")
print("============================================")

print(
    "Mean non-zero bands:",
    round(df["nonzero_bands"].mean(), 2)
)

print(
    "Median non-zero bands:",
    round(df["nonzero_bands"].median(), 2)
)

print(
    "Min non-zero bands:",
    df["nonzero_bands"].min()
)

print(
    "Max non-zero bands:",
    df["nonzero_bands"].max()
)

print()

print(
    "Samples >= 3 non-zero bands:",
    round((df["nonzero_bands"] >= 3).mean() * 100, 2),
    "%"
)

print(
    "Samples >= 5 non-zero bands:",
    round((df["nonzero_bands"] >= 5).mean() * 100, 2),
    "%"
)

print(
    "Samples >= 8 non-zero bands:",
    round((df["nonzero_bands"] >= 8).mean() * 100, 2),
    "%"
)

print()


# ============================================================
# 3. TOTAL FEATURE POWER
# ============================================================

df["total_band_power"] = (
    df[band_cols]
    .sum(axis=1)
)

print("============================================")
print("TOTAL FEATURE POWER")
print("============================================")

print(
    "Mean total power:",
    f"{df['total_band_power'].mean():.3e}"
)

print(
    "Median total power:",
    f"{df['total_band_power'].median():.3e}"
)

print(
    "Min total power:",
    f"{df['total_band_power'].min():.3e}"
)

print(
    "Max total power:",
    f"{df['total_band_power'].max():.3e}"
)

print()


# ============================================================
# 4. BASIC QUALITY
# ============================================================

df["good_bins_5"] = (
    df["bin_count"] >= 5
)

df["good_nonzero_3"] = (
    df["nonzero_bands"] >= 3
)

df["basic_good_sample"] = (
    df["good_bins_5"]
    &
    df["good_nonzero_3"]
)

good_percent = (
    df["basic_good_sample"].mean()
    * 100
)

good_count = int(
    df["basic_good_sample"].sum()
)

print("============================================")
print("BASIC QUALITY SUMMARY")
print("============================================")

print(
    "Samples passing basic check:",
    round(good_percent, 2),
    "%"
)

print(
    "Good samples:",
    good_count,
    "/",
    len(df)
)

print()


# ============================================================
# 5. DETECTED ANGLE DISTRIBUTION
# ============================================================

print("============================================")
print("DETECTED ANGLE DISTRIBUTION")
print("============================================")

for start in range(0, 360, 30):

    end = start + 30

    count = (
        (
            df["detected_angle"] >= start
        )
        &
        (
            df["detected_angle"] < end
        )
    ).sum()

    percent = (
        count
        / len(df)
        * 100
    )

    print(
        f"{start:3d} - {end - 1:3d} deg : "
        f"{count:4d} samples "
        f"({percent:6.2f}%)"
    )

print()


# ============================================================
# 6. BIN COUNT DISTRIBUTION
# ============================================================

print("============================================")
print("BIN COUNT DISTRIBUTION")
print("============================================")

ranges = [
    (3, 4),
    (5, 7),
    (8, 10),
    (11, 15),
    (16, 20),
    (21, 100)
]

for low, high in ranges:

    count = (
        (
            df["bin_count"] >= low
        )
        &
        (
            df["bin_count"] <= high
        )
    ).sum()

    percent = (
        count
        / len(df)
        * 100
    )

    print(
        f"{low:2d} - {high:3d} bins : "
        f"{count:4d} samples "
        f"({percent:6.2f}%)"
    )

print()


# ============================================================
# 7. NON-ZERO BAND DISTRIBUTION
# ============================================================

print("============================================")
print("NON-ZERO BAND DISTRIBUTION")
print("============================================")

for n in range(1, NUM_BANDS + 1):

    count = (
        df["nonzero_bands"] == n
    ).sum()

    if count == 0:
        continue

    percent = (
        count
        / len(df)
        * 100
    )

    print(
        f"{n:2d} non-zero bands : "
        f"{count:4d} samples "
        f"({percent:6.2f}%)"
    )

print()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("============================================")
print("FINAL SUMMARY")
print("============================================")

print(
    "Total samples        :",
    len(df)
)

print(
    "Mean bin count       :",
    round(df["bin_count"].mean(), 2)
)

print(
    "Mean non-zero bands  :",
    round(df["nonzero_bands"].mean(), 2)
)

print(
    "Basic good samples   :",
    good_count
)

print(
    "Basic good percentage:",
    round(good_percent, 2),
    "%"
)

print()

print("NOTE:")
print("Angle is only shown for reference.")
print("There is no expected-angle check in free-angle dataset collection.")