import csv
import statistics
import os
import glob

# ============================================================
# CẤU HÌNH NGƯỠNG ĐÁNH GIÁ (THEO BÀI BÁO)
# ============================================================
TOLERANCE_DEG = 15 
TARGET_DIR = "2sourceNOAI"

def circular_error(measured, true_angle):
    """Tính sai số vòng tròn giữa 2 góc (0-359)."""
    diff = abs(measured - true_angle) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def evaluate_frame_paper_standard(true_angles, detected_angles):
    """
    Đánh giá TP, FP, FN và MAE cho một khung hình dựa trên tiêu chuẩn ±15 độ.
    """
    TP = 0
    FP = 0
    FN = 0
    errors_in_frame = []
    
    unmatched_detected = list(detected_angles)
    
    for t_angle in true_angles:
        best_match_idx = -1
        min_err = float('inf')
        
        # Tìm góc phát hiện gần nhất với góc thực tế
        for idx, d_angle in enumerate(unmatched_detected):
            err = circular_error(d_angle, t_angle)
            if err < min_err:
                min_err = err
                best_match_idx = idx
                
        # Kiểm tra theo tiêu chuẩn ±15 độ của bài báo
        if best_match_idx != -1 and min_err <= TOLERANCE_DEG:
            TP += 1
            errors_in_frame.append(min_err)
            unmatched_detected.pop(best_match_idx)
        else:
            FN += 1
            
    FP += len(unmatched_detected)
    
    return TP, FP, FN, errors_in_frame

def process_csv_file(filepath):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    all_valid_errors = []
    
    total_frames = 0
    active_frames = 0
    true_sources_info = ""

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        true_angle_cols = [col for col in reader.fieldnames if col.startswith("true_angle_")]
        
        for row in reader:
            total_frames += 1
            detected_count = int(row["detected_count"])
            
            # Bỏ qua các khung hình Silence (VAD filtering)
            if detected_count == 0:
                continue
                
            active_frames += 1
            
            true_angles = [int(row[col]) for col in true_angle_cols if row[col] != ""]
            if not true_sources_info:
                true_sources_info = str(true_angles)
            
            raw_angles_str = row["raw_angles"]
            if raw_angles_str:
                detected_angles = [int(x) for x in raw_angles_str.split(";")]
            else:
                detected_angles = []
                
            tp, fp, fn, frame_errors = evaluate_frame_paper_standard(true_angles, detected_angles)
            
            total_TP += tp
            total_FP += fp
            total_FN += fn
            all_valid_errors.extend(frame_errors)
            
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mae = statistics.mean(all_valid_errors) if len(all_valid_errors) > 0 else 0.0

    return {
        "filename": os.path.basename(filepath),
        "true_sources": true_sources_info,
        "total_frames": total_frames,
        "active_frames": active_frames,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "errors": all_valid_errors,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mae": mae
    }

if __name__ == "__main__":
    print("================================================================================")
    print(" BATCH PAPER METRICS EVALUATOR (Threshold: +/- 15 deg, VAD Enabled)")
    print("================================================================================")
    
    csv_files = glob.glob(os.path.join(TARGET_DIR, "*.csv"))
    
    if not csv_files:
        print(f"Lỗi: Không tìm thấy file CSV nào trong thư mục '{TARGET_DIR}'.")
        exit()
        
    print(f"Đang quét {len(csv_files)} file CSV trong thư mục '{TARGET_DIR}'...\n")
    
    # Biến cộng dồn cho Overall Statistics
    agg_TP = 0
    agg_FP = 0
    agg_FN = 0
    agg_errors = []
    agg_total_frames = 0
    agg_active_frames = 0
    
    # In Header của Bảng
    print(f"{'File Name':<25} | {'Sources (deg)':<15} | {'Recall':<8} | {'Precision':<10} | {'F1':<6} | {'MAE':<6}")
    print("-" * 80)
    
    for file in sorted(csv_files):
        res = process_csv_file(file)
        
        # Cộng dồn dữ liệu
        agg_TP += res["TP"]
        agg_FP += res["FP"]
        agg_FN += res["FN"]
        agg_errors.extend(res["errors"])
        agg_total_frames += res["total_frames"]
        agg_active_frames += res["active_frames"]
        
        # Format kết quả để in ra bảng
        rec_str = f"{res['recall']:.1%}"
        prec_str = f"{res['precision']:.1%}"
        f1_str = f"{res['f1_score']:.2f}"
        mae_str = f"{res['mae']:.1f}"
        
        print(f"{res['filename']:<25} | {res['true_sources']:<15} | {rec_str:<8} | {prec_str:<10} | {f1_str:<6} | {mae_str:<6}")
        
    # Tính toán chỉ số Overall (Tổng hợp toàn bộ Dataset)
    overall_prec = agg_TP / (agg_TP + agg_FP) if (agg_TP + agg_FP) > 0 else 0.0
    overall_rec = agg_TP / (agg_TP + agg_FN) if (agg_TP + agg_FN) > 0 else 0.0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0
    overall_mae = statistics.mean(agg_errors) if len(agg_errors) > 0 else 0.0
    
    print("=" * 80)
    print("[TỔNG HỢP TOÀN BỘ DATASET 2 NGUỒN]")
    print(f"Tổng số file         : {len(csv_files)}")
    print(f"Tổng Frames thô      : {agg_total_frames}")
    print(f"Active Frames (VAD)  : {agg_active_frames}")
    print("-" * 40)
    print(f"OVERALL RECALL       : {overall_rec:.2%}")
    print(f"OVERALL PRECISION    : {overall_prec:.2%}")
    print(f"OVERALL F1-SCORE     : {overall_f1:.4f}")
    print(f"OVERALL MAE          : {overall_mae:.2f} deg")
    print("================================================================================")