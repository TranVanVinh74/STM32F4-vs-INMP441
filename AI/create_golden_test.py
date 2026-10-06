import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

WINDOW = 5
NUM_BANDS = 16

CSV_FILE = "final_test_raw/keep/speech/speech_11.csv"

band_cols = [f"band_{i}" for i in range(NUM_BANDS)]

model = keras.models.load_model("mlp_audio_temporal.keras")
scaler = joblib.load("scaler_temporal.pkl")

df = pd.read_csv(CSV_FILE)

# Lấy đúng 5 frame đầu tiên
window = df[band_cols].values[:WINDOW].astype(np.float32)

# Tạo 32 feature:
# 16 mean + 16 std
mean_feat = np.mean(window, axis=0)
std_feat = np.std(window, axis=0)

feature32 = np.concatenate([mean_feat, std_feat])

# Đưa qua đúng preprocessing như lúc train
x = np.log1p(feature32)
x_scaled = scaler.transform([x])

prob = model.predict(x_scaled, verbose=0)[0][0]

print("============================================")
print("GOLDEN TEST VECTOR")
print("============================================")

print()
print("Raw 32 features:")

for i, v in enumerate(feature32):
    print(f"{v:.9e}f,", end=" ")

    if (i + 1) % 4 == 0:
        print()

print()
print("Python probability =", prob)

if prob >= 0.5:
    print("Python class       = KEEP")
else:
    print("Python class       = NOISE")