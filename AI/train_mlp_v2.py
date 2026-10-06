import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Reproducible
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

# =========================
# MODEL
# =========================

model = keras.Sequential([
    layers.Input(shape=(16,)),

    layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    ),

    layers.Dropout(0.10),

    layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    ),

    layers.Dropout(0.10),

    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "mlp_audio_v2.keras",
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-5
)

# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# =========================
# TEST
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print()
print("============================================")
print("MLP V2 TEST")
print("============================================")
print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f} %")

print()
print("Epochs actually trained:", len(history.history["loss"]))
print("Best val_loss:", min(history.history["val_loss"]))
print("Best val_accuracy:", max(history.history["val_accuracy"]))