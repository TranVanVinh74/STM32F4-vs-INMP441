import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# LOAD DATA
# =========================

X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

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

model.summary()

# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# =========================
# TEST
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print()
print("============================================")
print("FINAL TEST")
print("============================================")
print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f} %")

# =========================
# SAVE MODEL
# =========================

model.save("mlp_audio.keras")

print()
print("Saved model: mlp_audio.keras")