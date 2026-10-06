# Biểu đồ cho báo cáo nghiên cứu khoa học

Chạy lại bằng PowerShell:
```powershell
cd D:\NCKH\AI
.\venv_ai\Scripts\python.exe .\plot_research_report.py
```
Nếu dữ liệu/mô hình đã đổi, chạy `report_model.py` trước rồi mới vẽ.
Mỗi hình có PNG 300 dpi để chèn Word, PDF và SVG dạng vector. File
`TOAN_BO_BIEU_DO.pdf` tập hợp cả 7 hình. Số liệu CSV kèm theo dùng để đối chiếu.

## Chú thích đề xuất

1. **Kiến trúc mô hình.** MLP nhận 32 đặc trưng gồm trung bình và độ lệch chuẩn
   của 16 dải trên cửa sổ 5 dòng CSV. Sau log1p và chuẩn hóa, dữ liệu đi qua các
   lớp Dense 32–16–8–1, tổng 1,729 tham số học được. Đầu ra sigmoid biểu diễn điểm P(KEEP).
2. **Quá trình huấn luyện.** Accuracy và binary cross-entropy trên train và
   validation trong 19 epoch. Checkpoint được chọn tại epoch 11
   theo validation loss nhỏ nhất (0.3550); validation accuracy
   tại epoch đó là 86.63%. Cả train và validation gốc có keyboard.
3. **Ma trận nhầm lẫn.** Kết quả trên 643 cửa sổ của tập kiểm thử cuối,
   đã loại keyboard. Hàng là nhãn thật, cột là dự đoán; phần trăm chuẩn hóa theo hàng.
   Có 14 cửa sổ NOISE bị giữ lại và 40 cửa sổ KEEP bị loại.
4. **Accuracy theo loại âm.** Tỷ lệ dự đoán nhị phân đúng đối với quạt, âm nhạc
   và tiếng nói. Đây không phải mô hình phân loại ba lớp; nhạc và tiếng nói đều mang nhãn KEEP.
5. **Precision, recall và F1.** Chỉ số cho từng lớp NOISE và KEEP trên tập kiểm thử
   cuối tại ngưỡng 0,5. Precision phản ánh độ đúng của dự đoán một lớp; recall phản
   ánh tỷ lệ nhận ra các mẫu thật thuộc lớp đó; F1 kết hợp precision và recall.
6. **ROC và precision–recall.** Lớp dương là KEEP; ROC-AUC = 0.9728,
   average precision (AP) = 0.9832. AP không được ghi thành diện tích hình thang PR-AUC.
   Dấu chấm biểu diễn ngưỡng 0,5 cố định, không chọn ngưỡng tối ưu từ tập test.
7. **Hai tập kiểm thử.** Accuracy và các chỉ số macro trên test nội bộ
   (970 cửa sổ) và kiểm thử cuối (643 cửa sổ), cùng loại keyboard.
   Macro là trung bình không trọng số của hai lớp. Hai tập khác nhau về dữ liệu,
   nên chênh lệch không chứng minh mô hình đã được cải thiện.

## Đoạn nhận xét dựa trên số liệu

Mô hình đạt accuracy 91.60% và macro-F1
0.9077 trên tập kiểm thử cuối. Recall của NOISE đạt
93.40%, còn recall của KEEP đạt
90.72%. Trên tập test nội bộ, accuracy đạt
79.59%. Sau epoch 11, validation loss không
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
