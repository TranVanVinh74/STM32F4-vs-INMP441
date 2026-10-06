"""Create Vietnamese publication figures from verified saved model results."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report_no_keyboard"
OUT = ROOT / "bieu_do_bao_cao"
COLORS = ["#2463A0", "#D07A24", "#30856B"]
NAMES = {"fan": "Quạt", "music": "Âm nhạc", "speech": "Tiếng nói"}
NOTE = "Kiểm thử đã loại keyboard • Mỗi mẫu là một cửa sổ 5 dòng CSV • Ngưỡng KEEP = 0,5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def read_results(name):
    df = pd.read_csv(SOURCE / f"{name}_predictions.csv")
    if df.empty or "keyboard" in set(df.sound_type.str.strip().str.lower()):
        raise ValueError(f"Invalid evaluation scope: {name}")
    if not np.isfinite(df.prob_keep).all() or not df.prob_keep.between(0, 1).all():
        raise ValueError(f"Invalid probabilities: {name}")
    if set(df.label.unique()) != {0, 1}:
        raise ValueError(f"Both classes are required: {name}")
    pred = (df.prob_keep >= 0.5).astype(int)
    if not np.array_equal(pred, df.prediction):
        raise ValueError(f"Predictions disagree with the 0.5 threshold: {name}")
    saved = json.loads((SOURCE / f"{name}_metrics.json").read_text())
    cm = confusion_matrix(df.label, pred, labels=[0, 1])
    if not np.array_equal(cm, saved["confusion_matrix"]) or len(df) != saved["windows"]:
        raise ValueError(f"Inconsistent report files: {name}")
    report = classification_report(df.label, pred, labels=[0, 1],
                                   target_names=["NOISE", "KEEP"],
                                   output_dict=True, zero_division=0)
    return df, report, cm


def save(fig, stem, pdf, note=NOTE):
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=9, color="#444444")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {stem}.png / .pdf / .svg")


def percent_axis(ax):
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Giá trị (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def labeled_bars(ax, x, values, **kwargs):
    bars = ax.bar(x, values, **kwargs)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=10)
    return bars


def architecture(config, pdf):
    layers = config["config"]["layers"]
    width = layers[0]["config"]["batch_shape"][-1]
    blocks = [("Đầu vào", f"{width} đặc trưng", "16 mean + 16 std")]
    params = []
    for layer in layers[1:]:
        if layer["class_name"] != "Dense":
            raise ValueError("Architecture plot expects a Dense-only MLP.")
        c = layer["config"]
        count = width * c["units"] + (c["units"] if c["use_bias"] else 0)
        params.append({"layer": c["name"], "input": width, "units": c["units"],
                       "activation": c["activation"], "parameters": count})
        blocks.append((f"Dense {c['units']}", c["activation"].capitalize(), f"{count:,} tham số"))
        width = c["units"]
    pd.DataFrame(params).to_csv(OUT / "tham_so_cac_lop.csv", index=False)
    total = sum(p["parameters"] for p in params)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")
    ax.set_title(f"Kiến trúc MLP phân loại âm thanh — {total:,} tham số học được", pad=18)
    for i, (title, activation, desc) in enumerate(blocks):
        x = 0.18 + i * 2.4
        ax.add_patch(FancyBboxPatch((x, 1.5), 2.0, 1.25,
                     boxstyle="round,pad=0.06", facecolor="#EAF1F8", edgecolor=COLORS[0]))
        ax.text(x + 1, 2.39, title, ha="center", weight="bold", fontsize=13)
        ax.text(x + 1, 2.04, activation, ha="center")
        ax.text(x + 1, 1.71, desc, ha="center", fontsize=10)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 2.32, 2.12), xytext=(x + 2.08, 2.12),
                        arrowprops={"arrowstyle": "->", "color": COLORS[0], "lw": 1.8})
    optimizer = config["compile_config"]["optimizer"]
    lr = optimizer["config"]["learning_rate"]
    ax.text(6, 0.91, "Tiền xử lý: 5 dòng → mean/std của 16 band → log1p → StandardScaler", ha="center")
    ax.text(6, 0.42, f"{optimizer['class_name']} • Learning rate = {lr:.3g} • Binary cross-entropy", ha="center")
    save(fig, "01_kien_truc_mo_hinh", pdf,
         "Đầu ra sigmoid: P(KEEP) ≥ 0,5 → KEEP; P(KEEP) < 0,5 → NOISE")
    return total


def training(history, pdf):
    best = history.loc[history.val_loss.idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy (%)", "Binary cross-entropy"]):
        scale = 100 if metric == "accuracy" else 1
        ax.plot(history.epoch, history[metric] * scale, color=COLORS[0], marker="o", ms=3, label="Train")
        ax.plot(history.epoch, history[f"val_{metric}"] * scale, color=COLORS[1], marker="s", ms=3, label="Validation")
        ax.axvline(best.epoch, color="#666666", ls="--", lw=1.2,
                   label=f"Checkpoint: epoch {int(best.epoch)}")
        ax.set(xlabel="Epoch", ylabel=label, title="(a) Độ chính xác" if metric == "accuracy" else "(b) Hàm mất mát")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        ax.set_xticks([1] + list(range(3, int(history.epoch.max()) + 1, 4)))
    save(fig, "02_qua_trinh_huan_luyen", pdf,
         "Lịch sử train/validation gốc có keyboard • Checkpoint chọn theo validation loss nhỏ nhất")
    return best


def confusion(cm, pdf):
    normalized = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(normalized, vmin=0, vmax=100, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({normalized[i,j]:.2f}%)", ha="center", va="center",
                    fontsize=18, color="white" if normalized[i,j] > 55 else "#222222")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NOISE", "KEEP"],
           yticklabels=["NOISE", "KEEP"], xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế",
           title="Ma trận nhầm lẫn trên tập kiểm thử cuối")
    fig.colorbar(im, ax=ax, label="Tỷ lệ theo từng hàng (%)", shrink=0.8)
    save(fig, "03_ma_tran_nham_lan", pdf)


def by_type(df, pdf):
    df = df.assign(correct=df.label == df.prediction)
    table = df.groupby("sound_type").agg(windows=("correct", "size"), correct=("correct", "sum"))
    table["accuracy_percent"] = 100 * table.correct / table.windows
    table.to_csv(OUT / "ket_qua_theo_loai_am.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 5.4))
    labels = [f"{NAMES.get(k, k)}\n(n = {r.windows:.0f})" for k, r in table.iterrows()]
    labeled_bars(ax, labels, table.accuracy_percent, color=COLORS, width=0.55)
    percent_axis(ax)
    ax.set(ylabel="Accuracy (%)", title="Độ chính xác theo loại âm — tập kiểm thử cuối")
    save(fig, "04_accuracy_theo_loai_am", pdf)


def class_metrics(report, pdf):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    x = np.arange(2)
    for i, (key, label) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1-score")]):
        vals = [report[c][key] * 100 for c in ["NOISE", "KEEP"]]
        labeled_bars(ax, x + (i - 1) * 0.24, vals, width=0.24, label=label, color=COLORS[i])
    ax.set_xticks(x, [f"{c}\n(n = {int(report[c]['support'])})" for c in ["NOISE", "KEEP"]])
    percent_axis(ax)
    ax.set_title("Precision, recall và F1-score — tập kiểm thử cuối", pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
    save(fig, "05_precision_recall_f1", pdf)


def curves(df, pdf):
    y, score = df.label.to_numpy(), df.prob_keep.to_numpy()
    fpr, tpr, roc_threshold = roc_curve(y, score)
    precision, recall, pr_threshold = precision_recall_curve(y, score)
    auc, ap = roc_auc_score(y, score), average_precision_score(y, score)
    report = classification_report(y, df.prediction, output_dict=True)
    cm = confusion_matrix(y, df.prediction, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS[0], label=f"ROC-AUC = {auc:.4f}", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Mốc ngẫu nhiên")
    ax.scatter(cm[0, 1] / cm[0].sum(), cm[1, 1] / cm[1].sum(), color=COLORS[1], zorder=4, label="Ngưỡng 0,5")
    ax.set(xlabel="False positive rate (NOISE → KEEP)", ylabel="True positive rate (recall KEEP)", title="(a) Đường ROC")
    ax = axes[1]
    ax.step(recall, precision, where="post", color=COLORS[0], label=f"Average precision = {ap:.4f}", lw=2)
    ax.axhline(y.mean(), ls="--", color="#888888", label=f"Tỷ lệ KEEP = {y.mean():.4f}")
    ax.scatter(report["1"]["recall"], report["1"]["precision"], color=COLORS[1], zorder=4, label="Ngưỡng 0,5")
    ax.set(xlabel="Recall (KEEP)", ylabel="Precision (KEEP)", title="(b) Đường precision–recall")
    for ax in axes:
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="lower left" if ax is axes[1] else "lower right")
    save(fig, "06_roc_precision_recall", pdf,
         "Tập kiểm thử cuối đã loại keyboard • Lớp dương: KEEP • AP là average precision")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_threshold}).to_csv(OUT / "roc_points.csv", index=False)
    pd.DataFrame({"recall": recall, "precision": precision,
                  "threshold": np.r_[pr_threshold, np.nan]}).to_csv(OUT / "pr_points.csv", index=False)
    return auc, ap


def comparison(internal, final, pdf):
    names = ["Accuracy", "Macro-precision", "Macro-recall", "Macro-F1"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    rows = []
    for i, (name, report) in enumerate([("Test nội bộ", internal), ("Kiểm thử cuối", final)]):
        vals = [report["accuracy"]] + [report["macro avg"][k] for k in ["precision", "recall", "f1-score"]]
        labeled_bars(ax, np.arange(4) + (i - 0.5) * 0.35, np.array(vals) * 100,
                     width=0.35, color=COLORS[i], label=name)
        rows.append({"dataset": name, **dict(zip(names, vals))})
    ax.set_xticks(np.arange(4), names)
    percent_axis(ax)
    ax.set_title("Kết quả trên hai tập kiểm thử", pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
    save(fig, "07_so_sanh_hai_tap_test", pdf)
    pd.DataFrame(rows).to_csv(OUT / "chi_so_tong_hop.csv", index=False, encoding="utf-8-sig")


def main():
    if not (SOURCE / "run_metadata.json").exists():
        raise SystemExit("Run .\\venv_ai\\Scripts\\python.exe .\\report_model.py first.")
    metadata = json.loads((SOURCE / "run_metadata.json").read_text(encoding="utf-8"))
    for name, digest in metadata["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Input changed: {name}. Run report_model.py again first.")
    internal_df, internal, _ = read_results("internal_test")
    final_df, final, cm = read_results("final_test")
    history = pd.read_csv(ROOT / "training_history_temporal.csv")
    with zipfile.ZipFile(ROOT / "mlp_audio_temporal.keras") as z:
        config = json.loads(z.read("config.json"))
    OUT.mkdir(exist_ok=True)
    with PdfPages(OUT / "TOAN_BO_BIEU_DO.pdf") as pdf:
        total = architecture(config, pdf)
        best = training(history, pdf)
        confusion(cm, pdf)
        by_type(final_df, pdf)
        class_metrics(final, pdf)
        auc, ap = curves(final_df, pdf)
        comparison(internal, final, pdf)
    pd.DataFrame(final).T.to_csv(OUT / "classification_report.csv", encoding="utf-8-sig")
    details = {
        "trainable_parameters": total, "epochs_run": len(history),
        "best_val_loss_epoch": int(best.epoch), "best_val_loss": float(best.val_loss),
        "val_accuracy_at_best_loss": float(best.val_accuracy),
        "final_windows": len(final_df), "internal_windows": len(internal_df),
        "final_accuracy": accuracy_score(final_df.label, final_df.prediction),
        "roc_auc_keep": auc, "average_precision_keep": ap,
    }
    (OUT / "thong_so_mo_hinh.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    captions = f"""# Biểu đồ cho báo cáo nghiên cứu khoa học

Chạy lại bằng PowerShell:
```powershell
cd D:\\NCKH\\AI
.\\venv_ai\\Scripts\\python.exe .\\plot_research_report.py
```
Nếu dữ liệu/mô hình đã đổi, chạy `report_model.py` trước rồi mới vẽ.
Mỗi hình có PNG 300 dpi để chèn Word, PDF và SVG dạng vector. File
`TOAN_BO_BIEU_DO.pdf` tập hợp cả 7 hình. Số liệu CSV kèm theo dùng để đối chiếu.

## Chú thích đề xuất

1. **Kiến trúc mô hình.** MLP nhận 32 đặc trưng gồm trung bình và độ lệch chuẩn
   của 16 dải trên cửa sổ 5 dòng CSV. Sau log1p và chuẩn hóa, dữ liệu đi qua các
   lớp Dense 32–16–8–1, tổng {total:,} tham số học được. Đầu ra sigmoid biểu diễn điểm P(KEEP).
2. **Quá trình huấn luyện.** Accuracy và binary cross-entropy trên train và
   validation trong {len(history)} epoch. Checkpoint được chọn tại epoch {int(best.epoch)}
   theo validation loss nhỏ nhất ({best.val_loss:.4f}); validation accuracy
   tại epoch đó là {best.val_accuracy * 100:.2f}%. Cả train và validation gốc có keyboard.
3. **Ma trận nhầm lẫn.** Kết quả trên {len(final_df)} cửa sổ của tập kiểm thử cuối,
   đã loại keyboard. Hàng là nhãn thật, cột là dự đoán; phần trăm chuẩn hóa theo hàng.
   Có {cm[0,1]} cửa sổ NOISE bị giữ lại và {cm[1,0]} cửa sổ KEEP bị loại.
4. **Accuracy theo loại âm.** Tỷ lệ dự đoán nhị phân đúng đối với quạt, âm nhạc
   và tiếng nói. Đây không phải mô hình phân loại ba lớp; nhạc và tiếng nói đều mang nhãn KEEP.
5. **Precision, recall và F1.** Chỉ số cho từng lớp NOISE và KEEP trên tập kiểm thử
   cuối tại ngưỡng 0,5. Precision phản ánh độ đúng của dự đoán một lớp; recall phản
   ánh tỷ lệ nhận ra các mẫu thật thuộc lớp đó; F1 kết hợp precision và recall.
6. **ROC và precision–recall.** Lớp dương là KEEP; ROC-AUC = {auc:.4f},
   average precision (AP) = {ap:.4f}. AP không được ghi thành diện tích hình thang PR-AUC.
   Dấu chấm biểu diễn ngưỡng 0,5 cố định, không chọn ngưỡng tối ưu từ tập test.
7. **Hai tập kiểm thử.** Accuracy và các chỉ số macro trên test nội bộ
   ({len(internal_df)} cửa sổ) và kiểm thử cuối ({len(final_df)} cửa sổ), cùng loại keyboard.
   Macro là trung bình không trọng số của hai lớp. Hai tập khác nhau về dữ liệu,
   nên chênh lệch không chứng minh mô hình đã được cải thiện.

## Đoạn nhận xét dựa trên số liệu

Mô hình đạt accuracy {final['accuracy'] * 100:.2f}% và macro-F1
{final['macro avg']['f1-score']:.4f} trên tập kiểm thử cuối. Recall của NOISE đạt
{final['NOISE']['recall'] * 100:.2f}%, còn recall của KEEP đạt
{final['KEEP']['recall'] * 100:.2f}%. Trên tập test nội bộ, accuracy đạt
{internal['accuracy'] * 100:.2f}%. Sau epoch {int(best.epoch)}, validation loss không
cải thiện trong lịch sử đã lưu trong khi train loss tiếp tục giảm; đây là dấu hiệu
mô hình bắt đầu khớp quá mức với dữ liệu train. Checkpoint dùng validation loss nhỏ nhất.

## Phạm vi diễn giải

- Keyboard được dùng bổ sung dữ liệu NOISE, có mặt trong train và validation gốc,
  nhưng không được tính trong mọi hình kiểm thử. Chưa có thí nghiệm đối chứng để
  kết luận việc thêm keyboard cải thiện mô hình.
- Kiểm thử cuối hiện chỉ có một file cho mỗi loại fan, music, speech; chưa có điều hòa.
  Kết quả chưa đại diện cho mọi thiết bị, môi trường hay tất cả âm có ích.
- Cửa sổ là 5 dòng liên tiếp, không đồng nghĩa 5 giây hay một đoạn dài cố định.
  Các cửa sổ cùng phiên có thể phụ thuộc nhau; không coi chúng là các thí nghiệm độc lập.
- Không gắn nhãn “blind test” nếu tập cuối từng được dùng để chọn mô hình hoặc tham số.
- P(KEEP) là đầu ra sigmoid; chưa có đánh giá hiệu chuẩn xác suất.
- Các biểu đồ sử dụng số liệu thực đã lưu, không làm mượt hay tạo lại lịch sử huấn luyện.
"""
    (OUT / "HUONG_DAN_VA_CHU_THICH.md").write_text(captions, encoding="utf-8")
    inputs = [ROOT / "plot_research_report.py", ROOT / "training_history_temporal.csv",
              ROOT / "mlp_audio_temporal.keras", SOURCE / "run_metadata.json"]
    inputs += [SOURCE / f"{name}_{kind}" for name in ["internal_test", "final_test"]
               for kind in ["predictions.csv", "metrics.json"]]
    (OUT / "figure_inputs_sha256.json").write_text(json.dumps({
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
