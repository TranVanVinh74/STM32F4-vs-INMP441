import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_history_temporal.csv")

# =========================
# ACCURACY
# =========================

plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["accuracy"], label="Train Accuracy")
plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

# =========================
# LOSS
# =========================

plt.figure(figsize=(8, 5))

plt.plot(df["epoch"], df["loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()

print("Saved:")
print("accuracy_curve.png")
print("loss_curve.png")