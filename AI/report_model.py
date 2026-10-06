"""Reproduce saved temporal-model results, excluding keyboard from evaluation."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "report_no_keyboard"
WINDOW = 5
BANDS = [f"band_{i}" for i in range(16)]
FEATURES = [f"mean_{i}" for i in range(16)] + [f"std_{i}" for i in range(16)]
EXCLUDED = {"keyboard"}


def predict(model, scaler, features):
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("Features must be finite and nonnegative.")
    x = scaler.transform(np.log1p(features))
    return model.predict(x, verbose=0).reshape(-1)


def evaluate_internal(model, scaler):
    df = pd.read_csv(ROOT / "test_temporal.csv")
    df = df.loc[~df["sound_type"].str.strip().str.lower().isin(EXCLUDED)].copy()
    probs = predict(model, scaler, df[FEATURES].to_numpy())
    out = df[["source_file", "sound_type", "label"]].copy()
    out["window_index"] = df.groupby("source_file").cumcount()
    out["prob_keep"] = probs
    return out


def evaluate_final(model, scaler):
    records = []
    for path in sorted((ROOT / "final_test_raw").rglob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Empty final test file: {path}")
        types = df["sound_type"].str.strip().str.lower()
        if types.nunique() != 1 or df["label"].nunique() != 1:
            raise ValueError(f"Expected one sound type and label per file: {path}")
        sound_type = types.iloc[0]
        if sound_type in EXCLUDED:
            continue
        raw = df[BANDS].to_numpy(dtype=np.float32)
        features = []
        for start in range(0, len(raw) - WINDOW + 1, WINDOW):
            window = raw[start:start + WINDOW]
            features.append(np.concatenate([window.mean(axis=0), window.std(axis=0)]))
        if not features:
            raise ValueError(f"Not enough rows for a window: {path}")
        probs = predict(model, scaler, features)
        records.append(pd.DataFrame({
            "source_file": path.relative_to(ROOT).as_posix(),
            "sound_type": sound_type,
            "label": int(df["label"].iloc[0]),
            "window_index": np.arange(len(probs)),
            "prob_keep": probs,
        }))
    if not records:
        raise ValueError("No eligible final test files.")
    return pd.concat(records, ignore_index=True)


def save_results(df, name):
    if df.empty or not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid or empty evaluation labels: {name}")
    df["prediction"] = (df["prob_keep"] >= 0.5).astype(int)
    df["correct"] = df["prediction"] == df["label"]
    df.to_csv(OUTPUT / f"{name}_predictions.csv", index=False)
    by_type = df.groupby("sound_type").agg(
        windows=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    by_type["accuracy_percent"] = by_type.pop("accuracy") * 100
    by_type.to_csv(OUTPUT / f"{name}_by_type.csv", index=False)
    by_file = df.groupby(["source_file", "sound_type"]).agg(
        windows=("correct", "size"), accuracy=("correct", "mean")
    ).reset_index()
    by_file["accuracy_percent"] = by_file.pop("accuracy") * 100
    by_file.to_csv(OUTPUT / f"{name}_by_file.csv", index=False)

    y, pred = df["label"], df["prediction"]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["NOISE", "KEEP"],
        digits=4, zero_division=0,
    )
    summary = {
        "windows": len(df),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix_labels": ["NOISE", "KEEP"],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["NOISE", "KEEP"],
            output_dict=True, zero_division=0,
        ),
    }
    (OUTPUT / f"{name}_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    text = (
        f"{name.upper()} - keyboard excluded from ALL evaluation metrics\n"
        f"Unit: non-overlapping windows of {WINDOW} consecutive CSV rows\n"
        f"Windows: {len(df)}\nAccuracy: {summary['accuracy'] * 100:.4f}%\n\n"
        f"{by_type.to_string(index=False)}\n\n{report}\n"
        f"Confusion matrix (rows=true, columns=predicted; NOISE, KEEP):\n{cm}\n"
    )
    (OUTPUT / f"{name}_results.txt").write_text(text, encoding="utf-8")
    print(text)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["NOISE", "KEEP"]).plot(
        ax=ax, values_format="d", colorbar=False
    )
    ax.set_title(f"{name.replace('_', ' ').title()} (excluding keyboard)")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(by_type["sound_type"], by_type["accuracy_percent"])
    ax.bar_label(bars, labels=[f"{x:.2f}%" for x in by_type["accuracy_percent"]], padding=3)
    ax.set(ylim=(0, 110), ylabel="Accuracy (%)", xlabel="Sound type",
           title=f"{name.replace('_', ' ').title()} (excluding keyboard)")
    ax.set_yticks(range(0, 101, 20))
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_accuracy_by_type.png", dpi=300)
    plt.close(fig)


def save_history():
    history = pd.read_csv(ROOT / "training_history_temporal.csv")
    for metric in ["accuracy", "loss"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["epoch"], history[metric], label="Train")
        ax.plot(history["epoch"], history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.title(),
               title=f"Original training history: {metric} (includes keyboard)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{metric}_curve.png", dpi=300)
        plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    model = keras.models.load_model(ROOT / "mlp_audio_temporal.keras", compile=False)
    scaler = joblib.load(ROOT / "scaler_temporal.pkl")
    save_results(evaluate_internal(model, scaler), "internal_test")
    save_results(evaluate_final(model, scaler), "final_test")
    save_history()
    inputs = [ROOT / name for name in [
        "mlp_audio_temporal.keras", "scaler_temporal.pkl", "test_temporal.csv",
        "training_history_temporal.csv", "report_model.py",
    ]] + sorted((ROOT / "final_test_raw").rglob("*.csv"))
    metadata = {
        "model": "mlp_audio_temporal.keras", "window_rows": WINDOW,
        "features": "16 means + 16 population standard deviations; log1p; saved StandardScaler",
        "keep_threshold": 0.5, "excluded_evaluation_types": sorted(EXCLUDED),
        "note": "Original training and validation include keyboard. No retraining performed. "
                "Final evaluation covers only fan, speech and music when using the current data. "
                "Folder naming alone does not establish that final data were never used for model selection.",
        "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in inputs},
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved reports to: {OUTPUT}")


if __name__ == "__main__":
    main()
