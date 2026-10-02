import serial
import csv
import statistics
import time
import os

# ============================================================
# CONFIG
# ============================================================

PORT = "COM4"
BAUD = 921600
NUM_SAMPLES = 100
DATA_DIR = "NOAI"

# ============================================================
# CIRCULAR ANGLE ERROR
# ============================================================

def circular_error(measured, true_angle):
    diff = abs(measured - true_angle) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

# ============================================================
# MATCH 2 DETECTED ANGLES WITH 2 TRUE ANGLES
# ============================================================

def match_two_sources(detected_1, detected_2, true_1, true_2):
    error_a1 = circular_error(detected_1, true_1)
    error_a2 = circular_error(detected_2, true_2)
    total_a = error_a1 + error_a2

    error_b1 = circular_error(detected_2, true_1)
    error_b2 = circular_error(detected_1, true_2)
    total_b = error_b1 + error_b2

    if total_a <= total_b:
        return detected_1, detected_2, error_a1, error_a2

    return detected_2, detected_1, error_b1, error_b2

# ============================================================
# CREATE FILE NAME
#
# Example:
# Run         = 1
# Source 1    = 0 deg
# Source 2    = 90 deg
# Distance    = 10 cm
#
# 01_00_90dis01.csv
#
# 20 cm  -> dis02
# 50 cm  -> dis05
# 100 cm -> dis10
# ============================================================

def create_filename(true_angle_1, true_angle_2, distance_cm):
    os.makedirs(DATA_DIR, exist_ok=True)

    distance_code = int(round(distance_cm / 10.0))
    run = 1

    while True:
        filename = (
            f"{run:02d}_"
            f"{true_angle_1:02d}_"
            f"{true_angle_2:02d}"
            f"dis{distance_code:02d}.csv"
        )

        full_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(full_path):
            return full_path, filename, run

        run += 1

# ============================================================
# INPUT
# ============================================================

true_angle_1 = int(input("Nhap goc that nguon 1 (0 - 359): "))
true_angle_2 = int(input("Nhap goc that nguon 2 (0 - 359): "))
distance_cm = float(input("Nhap khoang cach (cm), vi du 10: "))

if true_angle_1 < 0 or true_angle_1 > 359:
    print("Goc nguon 1 khong hop le.")
    exit()

if true_angle_2 < 0 or true_angle_2 > 359:
    print("Goc nguon 2 khong hop le.")
    exit()

if distance_cm <= 0:
    print("Khoang cach khong hop le.")
    exit()

# ============================================================
# CREATE OUTPUT FILE
# ============================================================

full_path, filename, run_number = create_filename(
    true_angle_1,
    true_angle_2,
    distance_cm
)

# ============================================================
# OPEN UART
# ============================================================

print()
print("============================================")
print("2-SOURCE DOA MEASUREMENT")
print("============================================")
print(f"Port         : {PORT}")
print(f"Baud         : {BAUD}")
print(f"Source 1     : {true_angle_1} deg")
print(f"Source 2     : {true_angle_2} deg")
print(f"Distance     : {distance_cm:.1f} cm")
print(f"Run          : {run_number}")
print(f"Frames       : {NUM_SAMPLES}")
print(f"Output       : {full_path}")
print()
print("Dang mo UART...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

time.sleep(2)
ser.reset_input_buffer()

# ============================================================
# COLLECT DATA
# ============================================================

results = []

print()
print("Bat dau thu du lieu...")
print()

while len(results) < NUM_SAMPLES:
    raw = ser.readline()

    if not raw:
        continue

    line = raw.decode(errors="ignore").strip()

    if not line.startswith("DOA,"):
        continue

    parts = line.split(",")

    if len(parts) < 2:
        continue

    try:
        peak_count = int(parts[1])
    except ValueError:
        continue

    detected_angles = []

    for i in range(peak_count):
        index = 2 + i

        if index >= len(parts):
            break

        try:
            angle = int(parts[index])
        except ValueError:
            continue

        if 0 <= angle <= 359:
            detected_angles.append(angle)

    sample_number = len(results) + 1

    detected_source_1 = None
    detected_source_2 = None
    error_source_1 = None
    error_source_2 = None
    mean_error = None

    if peak_count == 2 and len(detected_angles) == 2:
        detected_source_1, detected_source_2, error_source_1, error_source_2 = match_two_sources(
            detected_angles[0],
            detected_angles[1],
            true_angle_1,
            true_angle_2
        )

        mean_error = (error_source_1 + error_source_2) / 2.0

        print(
            f"{sample_number:3d}/{NUM_SAMPLES} | "
            f"DOA,2 | "
            f"S1 = {detected_source_1:3d} deg "
            f"(err {error_source_1:3d}) | "
            f"S2 = {detected_source_2:3d} deg "
            f"(err {error_source_2:3d})"
        )

    else:
        angle_text = ",".join(str(a) for a in detected_angles)

        print(
            f"{sample_number:3d}/{NUM_SAMPLES} | "
            f"Detected sources = {peak_count} | "
            f"Angles = [{angle_text}]"
        )

    results.append({
        "sample": sample_number,
        "true_angle_1": true_angle_1,
        "true_angle_2": true_angle_2,
        "distance_cm": distance_cm,
        "detected_count": peak_count,
        "raw_angles": ";".join(str(a) for a in detected_angles),
        "detected_angle_1": detected_source_1,
        "detected_angle_2": detected_source_2,
        "error_1": error_source_1,
        "error_2": error_source_2,
        "mean_error": mean_error
    })

ser.close()

# ============================================================
# STATISTICS
# ============================================================

valid_results = [
    r for r in results
    if r["detected_count"] == 2
    and r["error_1"] is not None
    and r["error_2"] is not None
]

count_0 = sum(r["detected_count"] == 0 for r in results)
count_1 = sum(r["detected_count"] == 1 for r in results)
count_2 = sum(r["detected_count"] == 2 for r in results)
count_3 = sum(r["detected_count"] == 3 for r in results)

two_source_rate = count_2 / len(results) * 100.0

print()
print("============================================")
print("2-SOURCE DOA RESULT")
print("============================================")
print(f"True source 1       : {true_angle_1} deg")
print(f"True source 2       : {true_angle_2} deg")
print(f"Distance            : {distance_cm:.1f} cm")
print(f"Total frames        : {len(results)}")
print()
print(f"Detected 0 source   : {count_0}")
print(f"Detected 1 source   : {count_1}")
print(f"Detected 2 sources  : {count_2}")
print(f"Detected 3 sources  : {count_3}")
print()
print(f"2-source detect rate: {two_source_rate:.2f}%")

if valid_results:
    errors_1 = [r["error_1"] for r in valid_results]
    errors_2 = [r["error_2"] for r in valid_results]
    all_errors = errors_1 + errors_2

    mean_error_1 = statistics.mean(errors_1)
    mean_error_2 = statistics.mean(errors_2)
    mean_error_all = statistics.mean(all_errors)

    median_error = statistics.median(all_errors)
    std_error = statistics.pstdev(all_errors)
    max_error = max(all_errors)

    within_5 = sum(e <= 5 for e in all_errors) / len(all_errors) * 100.0
    within_10 = sum(e <= 10 for e in all_errors) / len(all_errors) * 100.0
    within_20 = sum(e <= 20 for e in all_errors) / len(all_errors) * 100.0

    print()
    print("--------------------------------------------")
    print("ANGLE ERROR - FRAMES WITH 2 SOURCES")
    print("--------------------------------------------")
    print(f"Mean error source 1 : {mean_error_1:.2f} deg")
    print(f"Mean error source 2 : {mean_error_2:.2f} deg")
    print(f"Mean error overall  : {mean_error_all:.2f} deg")
    print(f"Median error        : {median_error:.2f} deg")
    print(f"Std deviation       : {std_error:.2f} deg")
    print(f"Maximum error       : {max_error:.2f} deg")
    print()
    print(f"Error <= 5 deg      : {within_5:.2f}%")
    print(f"Error <= 10 deg     : {within_10:.2f}%")
    print(f"Error <= 20 deg     : {within_20:.2f}%")
else:
    print()
    print("Khong co frame nao phat hien dung 2 nguon.")

# ============================================================
# SAVE CSV
# ============================================================

with open(
    full_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "sample",
        "true_angle_1",
        "true_angle_2",
        "distance_cm",
        "detected_count",
        "raw_angles",
        "detected_angle_1",
        "detected_angle_2",
        "error_1",
        "error_2",
        "mean_error"
    ])

    for r in results:
        writer.writerow([
            r["sample"],
            r["true_angle_1"],
            r["true_angle_2"],
            r["distance_cm"],
            r["detected_count"],
            r["raw_angles"],
            "" if r["detected_angle_1"] is None else r["detected_angle_1"],
            "" if r["detected_angle_2"] is None else r["detected_angle_2"],
            "" if r["error_1"] is None else r["error_1"],
            "" if r["error_2"] is None else r["error_2"],
            "" if r["mean_error"] is None else f"{r['mean_error']:.2f}"
        ])

print()
print("============================================")
print("Saved CSV:", full_path)
print("============================================")