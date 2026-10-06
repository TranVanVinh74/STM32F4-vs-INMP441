import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

X_train = np.load("X_train_temporal.npy")
X_val = np.load("X_val_temporal.npy")
X_test = np.load("X_test_temporal.npy")

y_train = np.load("y_train_temporal.npy")
y_val = np.load("y_val_temporal.npy")
y_test = np.load("y_test_temporal.npy")

model = keras.Sequential([
    layers.Input(shape=(32,)),
    layers.Dense(32, activation="relu"),
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
    "mlp_audio_temporal.keras",
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

history_df = pd.DataFrame(history.history)
history_df.index = history_df.index + 1
history_df.index.name = "epoch"

history_df.to_csv("training_history_temporal.csv")
print("Saved: training_history_temporal.csv")

best_model = keras.models.load_model("mlp_audio_temporal.keras")

test_loss, test_acc = best_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("============================================")
print("TEMPORAL MLP TEST")
print("============================================")
print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f} %")

print()
print("Epochs actually trained:", len(history.history["loss"]))
print("Best val_loss:", min(history.history["val_loss"]))
print("Best val_accuracy:", max(history.history["val_accuracy"]))