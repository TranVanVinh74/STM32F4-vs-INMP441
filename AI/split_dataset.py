import pandas as pd

INPUT_FILE = "dataset_merged.csv"

data = pd.read_csv(INPUT_FILE)

# Chia theo tên file/session
train_files = [
    # KEEP - music
    "music_01.csv",
    "music_02.csv",
    "music_03.csv",
    "music_04.csv",

    # KEEP - speech
    "speech_01.csv",
    "speech_02.csv",
    "speech_03.csv",
    "speech_04.csv",
    "speech_05.csv",
    "speech_08.csv",

    # NOISE - fan
    "fan_01.csv",
    "fan_02.csv",
    "fan_03.csv",
    "fan_05.csv",
    "fan_06.csv",
    "fan_07.csv",

    # NOISE - keyboard
    "keyboard_01.csv",
    "keyboard_02.csv",
    "keyboard_04.csv",
]

val_files = [
    "music_05.csv",

    "speech_06.csv",
    "speech_09.csv",

    "fan_04.csv",

    "keyboard_03.csv",
]

test_files = [
    "music_06.csv",

    "speech_07.csv",
    "speech_10.csv",

    "fan_08.csv",

    "keyboard_05.csv",
]


def match_file(path, filename):
    path = str(path).replace("\\", "/")
    return path.endswith(filename)


def select_by_files(df, file_list):
    mask = df["source_file"].apply(
        lambda x: any(match_file(x, f) for f in file_list)
    )
    return df[mask].copy()


train_df = select_by_files(data, train_files)
val_df = select_by_files(data, val_files)
test_df = select_by_files(data, test_files)

print("============================================")
print("DATASET SPLIT")
print("============================================")

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df),
]:
    print()
    print(name)
    print("Samples:", len(df))
    print("Files:", df["source_file"].nunique())
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())

    if "sound_type" in df.columns:
        print("Sound types:")
        print(df.groupby(["sound_type", "label"]).size())


train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print()
print("Saved:")
print("train.csv")
print("val.csv")
print("test.csv")