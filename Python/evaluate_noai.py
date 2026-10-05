import os
import csv
import glob
import statistics

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "2sourceNOAI")
REPORT_DIR = os.path.join(BASE_DIR, "NOAI_REPORT")

THRESHOLDS = [5, 10, 20]

# ============================================================
# HELPER
# ============================================================

def safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None

def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def percent(value, total):
    if total == 0:
        return 0.0
    return value / total * 100.0

# ============================================================
# READ ONE CSV FILE
# ============================================================

def read_measurement_file(filepath):
    rows = []

    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            detected_count = safe_int(row.get("detected_count"))
            true_angle_1 = safe_int(row.get("true_angle_1"))
            true_angle_2 = safe_int(row.get("true_angle_2"))
            distance_cm = safe_float(row.get("distance_cm"))

            detected_angle_1 = safe_int(row.get("detected_angle_1"))
            detected_angle_2 = safe_int(row.get("detected_angle_2"))

            error_1 = safe_float(row.get("error_1"))
            error_2 = safe_float(row.get("error_2"))

            rows.append({
                "detected_count": detected_count,
                "true_angle_1": true_angle_1,
                "true_angle_2": true_angle_2,
                "distance_cm": distance_cm,
                "detected_angle_1": detected_angle_1,
                "detected_angle_2": detected_angle_2,
                "error_1": error_1,
                "error_2": error_2
            })

    return rows

# ============================================================
# ANALYZE ONE FILE
# ============================================================

def analyze_file(filepath):
    rows = read_measurement_file(filepath)

    filename = os.path.basename(filepath)
    total_frames = len(rows)

    count_0 = sum(r["detected_count"] == 0 for r in rows)
    count_1 = sum(r["detected_count"] == 1 for r in rows)
    count_2 = sum(r["detected_count"] == 2 for r in rows)
    count_3 = sum(r["detected_count"] == 3 for r in rows)

    valid_rows = [
        r for r in rows
        if r["detected_count"] == 2
        and r["error_1"] is not None
        and r["error_2"] is not None
    ]

    errors_1 = [r["error_1"] for r in valid_rows]
    errors_2 = [r["error_2"] for r in valid_rows]
    all_errors = errors_1 + errors_2

    true_angle_1 = rows[0]["true_angle_1"] if rows else None
    true_angle_2 = rows[0]["true_angle_2"] if rows else None
    distance_cm = rows[0]["distance_cm"] if rows else None

    result = {
        "filename": filename,
        "true_angle_1": true_angle_1,
        "true_angle_2": true_angle_2,
        "distance_cm": distance_cm,
        "total_frames": total_frames,
        "detect_0": count_0,
        "detect_1": count_1,
        "detect_2": count_2,
        "detect_3": count_3,
        "two_source_detection_rate": percent(count_2, total_frames),
        "valid_two_source_frames": len(valid_rows)
    }

    if all_errors:
        result["mae_source_1"] = statistics.mean(errors_1)
        result["mae_source_2"] = statistics.mean(errors_2)
        result["mae_overall"] = statistics.mean(all_errors)
        result["median_error"] = statistics.median(all_errors)
        result["std_error"] = statistics.pstdev(all_errors)
        result["max_error"] = max(all_errors)
    else:
        result["mae_source_1"] = None
        result["mae_source_2"] = None
        result["mae_overall"] = None
        result["median_error"] = None
        result["std_error"] = None
        result["max_error"] = None

    for threshold in THRESHOLDS:
        source_correct = sum(
            error <= threshold
            for error in all_errors
        )

        result[f"source_error_le_{threshold}_rate"] = percent(
            source_correct,
            len(all_errors)
        )

        strict_correct_frames = sum(
            r["detected_count"] == 2
            and r["error_1"] is not None
            and r["error_2"] is not None
            and r["error_1"] <= threshold
            and r["error_2"] <= threshold
            for r in rows
        )

        result[f"strict_frame_accuracy_{threshold}"] = percent(
            strict_correct_frames,
            total_frames
        )

    return result, rows

# ============================================================
# SAVE PER-FILE SUMMARY
# ============================================================

def save_per_file_summary(results):
    output_file = os.path.join(
        REPORT_DIR,
        "summary_per_file.csv"
    )

    fields = [
        "filename",
        "true_angle_1",
        "true_angle_2",
        "distance_cm",
        "total_frames",
        "detect_0",
        "detect_1",
        "detect_2",
        "detect_3",
        "two_source_detection_rate",
        "valid_two_source_frames",
        "mae_source_1",
        "mae_source_2",
        "mae_overall",
        "median_error",
        "std_error",
        "max_error",
        "source_error_le_5_rate",
        "source_error_le_10_rate",
        "source_error_le_20_rate",
        "strict_frame_accuracy_5",
        "strict_frame_accuracy_10",
        "strict_frame_accuracy_20"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:
            row = result.copy()

            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = f"{value:.2f}"

            writer.writerow(row)

    return output_file

# ============================================================
# ANALYZE OVERALL
# ============================================================

def analyze_overall(all_rows):
    total_frames = len(all_rows)

    count_0 = sum(
        r["detected_count"] == 0
        for r in all_rows
    )

    count_1 = sum(
        r["detected_count"] == 1
        for r in all_rows
    )

    count_2 = sum(
        r["detected_count"] == 2
        for r in all_rows
    )

    count_3 = sum(
        r["detected_count"] == 3
        for r in all_rows
    )

    valid_rows = [
        r for r in all_rows
        if r["detected_count"] == 2
        and r["error_1"] is not None
        and r["error_2"] is not None
    ]

    errors_1 = [
        r["error_1"]
        for r in valid_rows
    ]

    errors_2 = [
        r["error_2"]
        for r in valid_rows
    ]

    all_errors = errors_1 + errors_2

    print()
    print("============================================================")
    print("OVERALL 2-SOURCE DOA RESULT - NO AI")
    print("============================================================")
    print(f"Total frames              : {total_frames}")
    print(f"Detected 0 source         : {count_0}")
    print(f"Detected 1 source         : {count_1}")
    print(f"Detected 2 sources        : {count_2}")
    print(f"Detected 3 sources        : {count_3}")
    print()

    detection_rate = percent(
        count_2,
        total_frames
    )

    print(
        f"2-source detection rate   : "
        f"{detection_rate:.2f}%"
    )

    if not all_errors:
        print("Khong co frame hop le de tinh sai so.")
        return

    mae_source_1 = statistics.mean(errors_1)
    mae_source_2 = statistics.mean(errors_2)
    mae_overall = statistics.mean(all_errors)
    median_error = statistics.median(all_errors)
    std_error = statistics.pstdev(all_errors)
    max_error = max(all_errors)

    print()
    print("------------------------------------------------------------")
    print("ANGLE ERROR - ONLY FRAMES DETECTING EXACTLY 2 SOURCES")
    print("------------------------------------------------------------")
    print(f"Valid 2-source frames     : {len(valid_rows)}")
    print(f"MAE source 1             : {mae_source_1:.2f} deg")
    print(f"MAE source 2             : {mae_source_2:.2f} deg")
    print(f"Overall MAE              : {mae_overall:.2f} deg")
    print(f"Median error             : {median_error:.2f} deg")
    print(f"Std deviation            : {std_error:.2f} deg")
    print(f"Maximum error            : {max_error:.2f} deg")

    print()
    print("------------------------------------------------------------")
    print("SOURCE-LEVEL ANGULAR ACCURACY")
    print("------------------------------------------------------------")

    for threshold in THRESHOLDS:
        correct = sum(
            e <= threshold
            for e in all_errors
        )

        rate = percent(
            correct,
            len(all_errors)
        )

        print(
            f"Error <= {threshold:2d} deg          : "
            f"{rate:.2f}%"
        )

    print()
    print("------------------------------------------------------------")
    print("STRICT FRAME ACCURACY")
    print("Frame dung khi tim DU 2 nguon va CA 2 goc deu dung nguong")
    print("------------------------------------------------------------")

    for threshold in THRESHOLDS:
        correct_frames = sum(
            r["detected_count"] == 2
            and r["error_1"] is not None
            and r["error_2"] is not None
            and r["error_1"] <= threshold
            and r["error_2"] <= threshold
            for r in all_rows
        )

        rate = percent(
            correct_frames,
            total_frames
        )

        print(
            f"Both errors <= {threshold:2d} deg    : "
            f"{rate:.2f}% "
            f"({correct_frames}/{total_frames})"
        )

# ============================================================
# PRINT PER-FILE RESULT
# ============================================================

def print_file_result(result):
    print(
        f"{result['filename']:25s} | "
        f"2-src = {result['two_source_detection_rate']:6.2f}% | "
        f"MAE = "
        f"{result['mae_overall'] if result['mae_overall'] is not None else 0:6.2f} deg | "
        f"Strict10 = {result['strict_frame_accuracy_10']:6.2f}%"
    )

# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.isdir(DATA_DIR):
        print("Khong tim thay thu muc:")
        print(DATA_DIR)
        return

    csv_files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.csv"
            )
        )
    )

    if not csv_files:
        print("Khong tim thay file CSV nao trong:")
        print(DATA_DIR)
        return

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    print("============================================================")
    print("NO-AI 2-SOURCE DOA EVALUATION")
    print("============================================================")
    print(f"Data folder : {DATA_DIR}")
    print(f"CSV files   : {len(csv_files)}")
    print()

    results = []
    all_rows = []

    for filepath in csv_files:
        try:
            result, rows = analyze_file(filepath)

            results.append(result)
            all_rows.extend(rows)

            print_file_result(result)

        except Exception as e:
            print(
                f"ERROR: {os.path.basename(filepath)} -> {e}"
            )

    if not results:
        print("Khong co file nao duoc xu ly thanh cong.")
        return

    report_file = save_per_file_summary(
        results
    )

    analyze_overall(
        all_rows
    )

    print()
    print("============================================================")
    print("REPORT SAVED")
    print("============================================================")
    print(report_file)

if __name__ == "__main__":
    main()