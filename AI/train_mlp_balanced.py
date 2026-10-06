import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

# =========================
# LOAD DATA
# =========================

X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

train_df = pd.read_csv("train.csv")

# =========================
# SESSION BALANCING
# =========================

file_counts = train_df["source_file"].value_counts()
num_files = len(file_counts)
num_samples = len(train_df)

sample_weights = np.zeros(num_samples, dtype=np.float32)

for i, source_file in enumerate(train_df["source_file"]):
    count = file_counts[source_file]

    sample_weights[i] = num_samples / (num_files * count)

print("============================================")
print("SESSION WEIGHTS")
print("============================================")

for source_file, count in file_counts.items():
    weight = num_samples / (num_files * count)
    filename = str(source_file).replace("\\", "/").split("/")[-1]

    print(
        f"{filename:18s} "
        f"samples={count:4d} "
        f"weight={weight:.4f}"
    )

# =========================
# MODEL
# =========================

model = keras.Sequential([
    layers.Input(shape=(16,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "mlp_audio_balanced.keras",
    monitor="val_loss",
    save_best_only=True
)

# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =========================
# TEST
# =========================

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("============================================")
print("SESSION BALANCED MLP TEST")
print("============================================")

print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f} %")

print()
print("Epochs actually trained:",
      len(history.history["loss"]))

print("Best val_loss:",
      min(history.history["val_loss"]))

print("Best val_accuracy:",
      max(history.history["val_accuracy"]))