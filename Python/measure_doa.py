import serial
import csv
import statistics
import time
import os
import itertools

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
# TỰ ĐỘNG GHÉP CẶP N GÓC ĐO ĐƯỢC VỚI N GÓC THỰC TẾ
# ============================================================

def match_n_sources(detected_angles, true_angles):
    best_permutation = None
    min_total_error = float('inf')
    best_errors = []

    # Sinh tất cả các hoán vị của góc đo được để tìm ra cách ghép khớp nhất
    for perm in itertools.permutations(detected_angles):
        current_errors = [circular_error(p, t) for p, t in zip(perm, true_angles)]
        total_error = sum(current_errors)
        
        if total_error < min_total_error:
            min_total_error = total_error
            best_permutation = perm
            best_errors = current_errors

    return list(best_permutation), best_errors

# ============================================================
# CREATE DYNAMIC FILE NAME
# ============================================================

def create_filename(num_sources, true_angles, distance_cm):
    # Tự động tạo tên thư mục theo số nguồn (ví dụ: 1sourceNOAI, 2sourceNOAI, 3sourceNOAI)
    data_dir = f"{num_sources}sourceNOAI"
    os.makedirs(data_dir, exist_ok=True)

    distance_code = int(round(distance_cm / 10.0))
    run = 1

    # Tạo chuỗi tên góc: VD 00_90_180
    angle_str = "_".join(f"{a:02d}" for a in true_angles)

    while True:
        filename = f"{run:02d}_{angle_str}dis{distance_code:02d}.csv"
        full_path = os.path.join(data_dir, filename)

        if not os.path.exists(full_path):
            return full_path, filename, run, data_dir

        run += 1

# ============================================================
# INPUT SECTION (DYNAMIC)
# ============================================================

print("============================================")
print("DOA MEASUREMENT TOOL (DYNAMIC SOURCES)")
print("============================================")

while True:
    try:
        num_sources = int(input("Nhap so luong nguon am thanh (1, 2, hoac 3): "))
        if 1 <= num_sources <= 3:
            break
        print("So luong nguon phai tu 1 den 3.")
    except ValueError:
        print("Vui long nhap mot so nguyen.")

true_angles = []
for i in range(num_sources):
    while True:
        try:
            angle = int(input(f"Nhap goc that nguon {i+1} (0 - 359): "))
            if 0 <= angle <= 359:
                true_angles.append(angle)
                break
            print("Goc khong hop le.")
        except ValueError:
            print("Vui long nhap mot so nguyen.")

# Sắp xếp các góc thực tế từ bé đến lớn để tạo tên file nhất quán
true_angles.sort()

while True:
    try:
        distance_cm = float(input("Nhap khoang cach (cm), vi du 10: "))
        if distance_cm > 0:
            break
        print("Khoang cach khong hop le.")
    except ValueError:
        print("Vui long nhap mot so.")

# ============================================================
# CREATE OUTPUT FILE
# ============================================================

full_path, filename, run_number, current_dir = create_filename(
    num_sources,
    true_angles,
    distance_cm
)

# ============================================================
# OPEN UART
# ============================================================

print()
print("============================================")
print(f"{num_sources}-SOURCE DOA MEASUREMENT STARTING...")
print("============================================")
print(f"Port         : {PORT}")
print(f"Baud         : {BAUD}")
for i, a in enumerate(true_angles):
    print(f"Source {i+1}     : {a} deg")
print(f"Distance     : {distance_cm:.1f} cm")
print(f"Run          : {run_number}")
print(f"Frames       : {NUM_SAMPLES}")
print(f"Output Dir   : {current_dir}/")
print(f"Output File  : {filename}")
print()
print("Dang mo UART...")

ser = serial.Serial(PORT, BAUD, timeout=1)
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
            if 0 <= angle <= 359:
                detected_angles.append(angle)
        except ValueError:
            continue

    sample_number = len(results) + 1

    matched_detected = [None] * num_sources
    errors = [None] * num_sources
    mean_error = None

    # Chỉ tính error nếu số nguồn phát hiện được bằng ĐÚNG số nguồn thực tế
    if peak_count == num_sources and len(detected_angles) == num_sources:
        matched_detected, errors = match_n_sources(detected_angles, true_angles)
        mean_error = sum(errors) / float(num_sources)

        log_str = f"{sample_number:3d}/{NUM_SAMPLES} | DOA,{num_sources} | "
        for i in range(num_sources):
            log_str += f"S{i+1}={matched_detected[i]:3d} (e:{errors[i]:3d}) "
        print(log_str)
    else:
        angle_text = ",".join(str(a) for a in detected_angles)
        print(f"{sample_number:3d}/{NUM_SAMPLES} | Detected: {peak_count} | Angles: [{angle_text}]")

    results.append({
        "sample": sample_number,
        "distance_cm": distance_cm,
        "detected_count": peak_count,
        "raw_angles": ";".join(str(a) for a in detected_angles),
        "matched_detected": matched_detected,
        "errors": errors,
        "mean_error": mean_error
    })

ser.close()

# ============================================================
# STATISTICS
# ============================================================

valid_results = [r for r in results if r["detected_count"] == num_sources and all(e is not None for e in r["errors"])]

counts = {0: 0, 1: 0, 2: 0, 3: 0, "more": 0}
for r in results:
    c = r["detected_count"]
    if c in counts:
        counts[c] += 1
    else:
        counts["more"] += 1

success_rate = counts.get(num_sources, 0) / len(results) * 100.0

print()
print("============================================")
print(f"RESULT SUMMARY ({num_sources} SOURCES)")
print("============================================")
print(f"True sources : {true_angles} deg")
print(f"Distance     : {distance_cm:.1f} cm")
print(f"Total frames : {len(results)}")
print()
for i in range(4):
    print(f"Detected {i} source(s) : {counts[i]}")
if counts["more"] > 0:
    print(f"Detected >3 sources : {counts['more']}")
print()
print(f"Exact match detect rate: {success_rate:.2f}%")

if valid_results:
    all_errors = []
    source_errors = [[] for _ in range(num_sources)]
    
    for r in valid_results:
        for i in range(num_sources):
            source_errors[i].append(r["errors"][i])
            all_errors.append(r["errors"][i])

    print()
    print("--------------------------------------------")
    print(f"ANGLE ERROR (ONLY FRAMES DETECTED EXACTLY {num_sources} SOURCES)")
    print("--------------------------------------------")
    
    for i in range(num_sources):
        mean_err = statistics.mean(source_errors[i])
        print(f"Mean error Source {i+1} ({true_angles[i]} deg) : {mean_err:.2f} deg")
    
    mean_error_all = statistics.mean(all_errors)
    median_error = statistics.median(all_errors)
    std_error = statistics.pstdev(all_errors) if len(all_errors) > 1 else 0.0
    max_error = max(all_errors)

    within_5 = sum(e <= 5 for e in all_errors) / len(all_errors) * 100.0
    within_10 = sum(e <= 10 for e in all_errors) / len(all_errors) * 100.0
    within_20 = sum(e <= 20 for e in all_errors) / len(all_errors) * 100.0

    print(f"Overall Mean error       : {mean_error_all:.2f} deg")
    print(f"Overall Median error     : {median_error:.2f} deg")
    print(f"Overall Std deviation    : {std_error:.2f} deg")
    print(f"Overall Maximum error    : {max_error:.2f} deg")
    print()
    print(f"Frames with Error <= 5 deg  : {within_5:.2f}%")
    print(f"Frames with Error <= 10 deg : {within_10:.2f}%")
    print(f"Frames with Error <= 20 deg : {within_20:.2f}%")
else:
    print("\nKhong co frame nao phat hien dung so luong nguon yeu cau.")

# ============================================================
# SAVE CSV DYNAMICALLY
# ============================================================

# Tạo header động
csv_headers = ["sample", "distance_cm", "detected_count", "raw_angles"]
for i in range(num_sources):
    csv_headers.append(f"true_angle_{i+1}")
for i in range(num_sources):
    csv_headers.append(f"detected_angle_{i+1}")
for i in range(num_sources):
    csv_headers.append(f"error_{i+1}")
csv_headers.append("mean_error")

with open(full_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

    for r in results:
        row = [
            r["sample"],
            r["distance_cm"],
            r["detected_count"],
            r["raw_angles"]
        ]
        # Điền true angles
        for a in true_angles:
            row.append(a)
        
        # Điền detected angles (nếu match)
        for i in range(num_sources):
            val = r["matched_detected"][i]
            row.append("" if val is None else val)
            
        # Điền errors (nếu match)
        for i in range(num_sources):
            val = r["errors"][i]
            row.append("" if val is None else val)
            
        # Điền mean error
        me = r["mean_error"]
        row.append("" if me is None else f"{me:.2f}")

        writer.writerow(row)

print()
print("============================================")
print("Saved CSV:", full_path)
print("============================================")