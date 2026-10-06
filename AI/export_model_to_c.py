import numpy as np
import joblib
from tensorflow import keras

MODEL_FILE = "mlp_audio_temporal.keras"
SCALER_FILE = "scaler_temporal.pkl"
OUTPUT_FILE = "ai_model_data.h"

model = keras.models.load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, keras.layers.Dense)
]

print("============================================")
print("MODEL EXPORT")
print("============================================")

print("Scaler mean shape :", scaler.mean_.shape)
print("Scaler scale shape:", scaler.scale_.shape)
print("Dense layers      :", len(dense_layers))

for i, layer in enumerate(dense_layers):
    w, b = layer.get_weights()

    print(
        f"Layer {i+1}: "
        f"W={w.shape}, "
        f"B={b.shape}"
    )


def write_1d_array(f, name, arr):
    arr = np.asarray(arr).flatten()

    f.write(
        f"static const float {name}[{len(arr)}] = {{\n"
    )

    for i, value in enumerate(arr):
        f.write(f"    {value:.9e}f")

        if i != len(arr) - 1:
            f.write(",")

        if (i + 1) % 4 == 0:
            f.write("\n")
        else:
            f.write(" ")

    f.write("\n};\n\n")


def write_2d_array(f, name, arr):
    arr = np.asarray(arr)

    rows, cols = arr.shape

    f.write(
        f"static const float "
        f"{name}[{rows}][{cols}] = {{\n"
    )

    for r in range(rows):
        f.write("    {")

        for c in range(cols):
            f.write(f"{arr[r, c]:.9e}f")

            if c != cols - 1:
                f.write(", ")

        f.write("}")

        if r != rows - 1:
            f.write(",")

        f.write("\n")

    f.write("};\n\n")


with open(OUTPUT_FILE, "w") as f:

    f.write("#ifndef AI_MODEL_DATA_H\n")
    f.write("#define AI_MODEL_DATA_H\n\n")

    f.write("#define AI_INPUT_SIZE 32\n")
    f.write("#define AI_H1_SIZE 32\n")
    f.write("#define AI_H2_SIZE 16\n")
    f.write("#define AI_H3_SIZE 8\n")
    f.write("#define AI_OUTPUT_SIZE 1\n\n")

    # =========================
    # SCALER
    # =========================

    write_1d_array(
        f,
        "ai_scaler_mean",
        scaler.mean_
    )

    write_1d_array(
        f,
        "ai_scaler_scale",
        scaler.scale_
    )

    # =========================
    # MODEL WEIGHTS
    # =========================

    for i, layer in enumerate(dense_layers):
        w, b = layer.get_weights()

        write_2d_array(
            f,
            f"ai_w{i+1}",
            w
        )

        write_1d_array(
            f,
            f"ai_b{i+1}",
            b
        )

    f.write("#endif\n")


print()
print("Export completed.")
print("Generated:", OUTPUT_FILE)