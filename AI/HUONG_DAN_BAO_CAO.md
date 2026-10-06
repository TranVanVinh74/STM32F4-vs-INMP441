# Chạy lại kết quả mô hình cho báo cáo

Mở PowerShell tại thư mục dự án rồi chạy:

```powershell
cd D:\NCKH\AI
.\venv_ai\Scripts\python.exe .\report_model.py
```

Không cần kích hoạt môi trường ảo. Lệnh dùng mô hình `mlp_audio_temporal.keras`
và bộ chuẩn hóa `scaler_temporal.pkl` đã lưu, không huấn luyện lại.
Kết quả được tạo/cập nhật trong `report_no_keyboard`.

## Phạm vi keyboard

- Giữ nguyên mô hình đã huấn luyện với keyboard mang nhãn NOISE.
- Loại keyboard trước khi tính **mọi** chỉ số của test nội bộ và final test:
  accuracy, precision, recall, F1, ma trận nhầm lẫn và biểu đồ.
- Dữ liệu validation trong lần huấn luyện gốc cũng có keyboard. Vì vậy không
  mô tả thí nghiệm cũ là “keyboard chỉ xuất hiện trong tập train”.
- Hai biểu đồ accuracy/loss lấy từ lịch sử huấn luyện cũ; train và validation
  trên các biểu đồ này vẫn bao gồm keyboard.
- Vai trò bổ sung độ đa dạng của NOISE là mục đích sử dụng keyboard. Chưa có
  thí nghiệm đối chứng có/không keyboard để kết luận keyboard cải thiện kết quả.

## Mô hình và đơn vị đánh giá

- MLP: 32 đầu vào → Dense 32 ReLU → Dense 16 ReLU → Dense 8 ReLU → sigmoid 1.
- Một mẫu đánh giá là nhóm 5 dòng CSV liên tiếp, không chồng lấp, trong cùng file.
  Các dòng dư cuối file không đủ một nhóm bị bỏ qua.
- Đặc trưng: trung bình 16 band và độ lệch chuẩn tổng thể 16 band, sau đó
  `log1p` và StandardScaler đã lưu từ quá trình huấn luyện.
- Ngưỡng KEEP: xác suất >= 0,5; NOISE = 0, KEEP = 1.
- Các nhóm 5 dòng không được diễn giải thành khoảng thời gian cố định khi chưa
  xác minh nhịp lấy mẫu; CSV hiện có cũng chưa chứng minh các dòng cùng một nguồn vật lý.

## Kết quả chạy lại

| Tập đánh giá, đã loại keyboard | Số cửa sổ | Accuracy |
|---|---:|---:|
| Test nội bộ (`test_temporal.csv`) | 970 | 79,59% |
| Final test (`final_test_raw`) | 643 | 91,60% |

| Loại âm trong final test | Số cửa sổ | Số đúng | Accuracy |
|---|---:|---:|---:|
| fan | 212 | 198 | 93,40% |
| music | 223 | 186 | 83,41% |
| speech | 208 | 205 | 98,56% |

Final test: macro-F1 = 0,9077; KEEP precision = 0,9654,
KEEP recall = 0,9072; NOISE recall = 0,9340.

Ma trận nhầm lẫn final test, hàng là nhãn thật, cột là dự đoán:

| | Dự đoán NOISE | Dự đoán KEEP |
|---|---:|---:|
| Thật NOISE | 198 | 14 |
| Thật KEEP | 40 | 391 |

Accuracy tổng được tính trên tất cả cửa sổ, không phải trung bình cộng accuracy
của ba loại âm. Số liệu hiện tại chỉ đánh giá fan, music và speech; chưa có
dữ liệu final test điều hòa và chưa chứng minh khả năng giữ tất cả loại âm khác.
Tên thư mục final test không tự chứng minh tập này chưa từng được dùng để chọn
mô hình; chỉ gọi là blind test nếu quy trình thực nghiệm thực sự đáp ứng điều đó.

## File dùng trong báo cáo

- `report_no_keyboard/final_test_results.txt`: các chỉ số final test.
- `report_no_keyboard/internal_test_results.txt`: các chỉ số test nội bộ.
- `report_no_keyboard/final_test_by_type.csv`: bảng theo loại âm.
- `report_no_keyboard/final_test_accuracy_by_type.png`: biểu đồ theo loại âm.
- `report_no_keyboard/final_test_confusion_matrix.png`: ma trận nhầm lẫn.
- `report_no_keyboard/accuracy_curve.png`, `loss_curve.png`: lịch sử huấn luyện cũ.
- Các file `*_predictions.csv`: dự đoán theo từng cửa sổ để đối chiếu.
- `report_no_keyboard/run_metadata.json`: cấu hình và SHA-256 của các file đầu vào.

Các script cũ `evaluate_final_test.py`, `plot_final_results.py` và
`evaluate_temporal_by_file.py` vẫn bao gồm keyboard. Dùng `report_model.py` cho
phạm vi báo cáo này. Chạy lại `train_mlp_temporal.py` sẽ huấn luyện lại và ghi đè
mô hình cùng lịch sử; việc đó không cần thiết để tái đánh giá mô hình hiện có.

Đoạn mô tả có thể dùng:

> Mô hình phân loại nhị phân KEEP/NOISE được huấn luyện với dữ liệu tiếng nói,
> âm nhạc, quạt và bàn phím; tiếng bàn phím được dùng làm dữ liệu bổ sung cho
> lớp NOISE và có mặt trong tập validation của quá trình huấn luyện. Phạm vi
> đánh giá trong báo cáo gồm tiếng nói, âm nhạc và quạt, loại bàn phím khỏi mọi
> chỉ số kiểm thử. Trên tập final test gồm 643 cửa sổ, mỗi cửa sổ gồm 5 dòng
> đặc trưng liên tiếp, mô hình đạt accuracy 91,60% và macro-F1 0,9077. Trên tập
> test nội bộ với cùng phạm vi loại âm, accuracy đạt 79,59% trên 970 cửa sổ.
