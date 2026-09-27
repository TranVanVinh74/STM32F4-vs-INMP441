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


# ============================================================
# CIRCULAR ANGLE ERROR
# ============================================================

def circular_error(measured, true_angle):
    diff = abs(measured - true_angle) % 360

    if diff > 180:
        diff = 360 - diff

    return diff


# ============================================================
# CREATE FILE NAME
#
# Example:
# angle = 0 deg
# distance = 0.5 m
#
# 000deg05dis01.csv
# 000deg05dis02.csv
# ...
# ============================================================

def create_filename(true_angle, distance):

    # 0.5 m -> 05
    # 1.0 m -> 10
    # 1.5 m -> 15
    # 2.0 m -> 20
    distance_code = int(round(distance * 10))

    run = 1

    while True:

        filename = (
            f"{true_angle:03d}deg"
            f"{distance_code:02d}dis"
            f"{run:02d}.csv"
        )

        # Neu file chua ton tai thi dung ten nay
        if not os.path.exists(filename):
            return filename, run

        run += 1


# ============================================================
# INPUT
# ============================================================

true_angle = int(
    input("Nhap goc that (0 - 359): ")
)

distance = float(
    input("Nhap khoang cach (m), vi du 0.5: ")
)


if true_angle < 0 or true_angle > 359:
    print("Goc khong hop le.")
    exit()

if distance <= 0:
    print("Khoang cach khong hop le.")
    exit()


# ============================================================
# CREATE OUTPUT FILE NAME
# ============================================================

filename, run_number = create_filename(
    true_angle,
    distance
)


# ============================================================
# OPEN UART
# ============================================================

print()
print("============================================")
print("DOA MEASUREMENT")
print("============================================")

print(f"Port       : {PORT}")
print(f"Baud       : {BAUD}")
print(f"True angle : {true_angle} deg")
print(f"Distance   : {distance:.2f} m")
print(f"Run        : {run_number}")
print(f"Samples    : {NUM_SAMPLES}")
print(f"Output     : {filename}")

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

    line = raw.decode(
        errors="ignore"
    ).strip()


    # ========================================================
    # ONLY ACCEPT DOA MESSAGE
    # ========================================================

    if not line.startswith("DOA,"):
        continue


    parts = line.split(",")

    if len(parts) < 2:
        continue


    try:
        peak_count = int(parts[1])
    except ValueError:
        continue


    # ========================================================
    # SINGLE SOURCE TEST
    #
    # Chi lay frame co dung 1 source
    # ========================================================

    if peak_count != 1:
        continue


    if len(parts) < 3:
        continue


    try:
        detected_angle = int(parts[2])
    except ValueError:
        continue


    if detected_angle < 0 or detected_angle > 359:
        continue


    # ========================================================
    # CALCULATE ERROR
    # ========================================================

    error = circular_error(
        detected_angle,
        true_angle
    )


    results.append({
        "sample": len(results) + 1,
        "true_angle": true_angle,
        "distance_m": distance,
        "detected_angle": detected_angle,
        "error": error
    })


    print(
        f"{len(results):3d}/{NUM_SAMPLES} | "
        f"True = {true_angle:3d} deg | "
        f"Detected = {detected_angle:3d} deg | "
        f"Error = {error:3d} deg"
    )


ser.close()


# ============================================================
# STATISTICS
# ============================================================

errors = [
    r["error"]
    for r in results
]


mean_error = statistics.mean(errors)

median_error = statistics.median(errors)

max_error = max(errors)

std_error = statistics.pstdev(errors)


within_5 = (
    sum(e <= 5 for e in errors)
    / len(errors)
    * 100
)

within_10 = (
    sum(e <= 10 for e in errors)
    / len(errors)
    * 100
)

within_20 = (
    sum(e <= 20 for e in errors)
    / len(errors)
    * 100
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("============================================")
print("DOA MEASUREMENT RESULT")
print("============================================")

print(f"True angle        : {true_angle} deg")
print(f"Distance          : {distance:.2f} m")
print(f"Run number        : {run_number}")
print(f"Number of samples : {len(results)}")

print()

print(f"Mean abs error    : {mean_error:.2f} deg")
print(f"Median error      : {median_error:.2f} deg")
print(f"Std deviation     : {std_error:.2f} deg")
print(f"Maximum error     : {max_error:.2f} deg")

print()

print(f"Error <= 5 deg    : {within_5:.2f}%")
print(f"Error <= 10 deg   : {within_10:.2f}%")
print(f"Error <= 20 deg   : {within_20:.2f}%")


# ============================================================
# SAVE CSV
# ============================================================

with open(
    filename,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    # HEADER
    writer.writerow([
        "sample",
        "true_angle",
        "distance_m",
        "detected_angle",
        "error"
    ])


    # DATA
    for r in results:

        writer.writerow([
            r["sample"],
            r["true_angle"],
            r["distance_m"],
            r["detected_angle"],
            r["error"]
        ])


print()
print("============================================")
print("Saved CSV:", filename)
print("============================================")