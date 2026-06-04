# Bare Minimum Specification

## 1. Mục tiêu cốt lõi
*   **Bài toán:** Phân loại nhị phân (Binary Classification).
*   **Target Variable:** `match` (1: Cả hai cùng đồng ý, 0: Ngược lại).
*   **Chiến lược lấy mẫu:** "Cân bằng khi học (Train), Khách quan khi thi (Test)" để đối phó với dữ liệu lệch.
*   **Metric đánh giá chính:** Accuracy, Confusion Matrix, Precision, Recall, F1-score (đặc biệt quan trọng do mất cân bằng lớp) và ROC-AUC.
*   **Biến phụ trợ:** `dec` (Quyết định cá nhân của chủ thể).

## 2. Dữ liệu đầu vào "Sống còn" (Features)
Để mô hình chạy được mức tối thiểu, cần tập trung vào các nhóm này:
*   **Cá nhân (Subject):** `gender`, `age`, `age_o` (tuổi đối phương), `race`, `imprace` (tầm quan trọng của sắc tộc), `imprelig` (tầm quan trọng của tôn giáo).
*   **Sở thích cá nhân (Hobbies - 17 biến):** `sports`, `tvsports`, `exercise`, `dining`, `museums`, `art`, `hiking`, `gaming`, `clubbing`, `reading`, `tv`, `theater`, `movies`, `concerts`, `music`, `shopping`, `yoga`.
*   **Tiêu chí chọn bạn đời (Preferences - Time 1):** `attr1_1` (Hấp dẫn), `sinc1_1` (Chân thành), `intel1_1` (Thông minh), `fun1_1` (Vui vẻ), `amb1_1` (Tham vọng), `shar1_1` (Sở thích chung).
*   **Đánh giá đối phương (Partner Scorecard):** `attr`, `sinc`, `intel`, `fun`, `amb`, `shar` (Điểm số đối phương chấm cho mình sau cuộc gặp).
*   **Biến giả thuyết:** `condtn` (1: Dưới 10 lựa chọn, 2: Trên 20 lựa chọn).

## 3. Giả thuyết nghiên cứu (Research Questions)
*   **H1 - Nghịch lý lựa chọn:** Liệu tỷ lệ `match` ở `condtn=2` có thấp hơn đáng kể so với `condtn=1` không?
*   **H2 - Yếu tố quyết định:** Thuộc tính nào (ví dụ: `attr` vs `intel`) có trọng số cao nhất trong việc dự đoán `match`?
*   **H3 - Chênh lệch tuổi tác:** Sự đồng điệu về lứa tuổi (`age_diff`) có phải là một yếu tố thúc đẩy việc kết đôi?

## 4. Tiêu chuẩn Tiền xử lý (Data Preparation Standard)

### Chuẩn bị dữ liệu phải theo thứ tự:
1. **Missing Values:** Xác định chiến lược (entity→impute, relationship→delete nếu quá thiếu)
2. **Đồng bộ hóa:** Nếu xóa entity, phải xóa relationship liên quan (referential integrity)
3. **Feature Engineering:** Tính biến mới (age_diff) TRƯỚC khi scale
4. **Scaling:** MinMax cho thang giới hạn, Standard cho liên tục
5. **Outlier:** IQR clip thay vì drop
6. **Dimensionality Reduction:** Gộp có ý nghĩa (dựa correlation) thay vì loại bỏ

### Những "trap" cần tránh:
- ❌ Scale age → rồi tính age_diff (sai: sẽ dùng Z-score age)
- ❌ Xóa user mà không xóa interaction của họ (data corruption)
- ❌ Impute relationship data bằng trung bình (tạo dữ liệu giả)
- ❌ Một cách scaling cho tất cả biến (cần phân biệt thang đo)

## 5. Chiến lược Tối ưu hóa Nâng cao (Advanced Optimization)
*   **Vòng lặp Insight:** Sử dụng mô hình phi tuyến (Gradient Boosting) để phát hiện các quy luật ngầm, sau đó "suy ngược" để cải tiến dữ liệu.
*   **Đặc trưng tinh lọc (Distilled Features):** Chuyển hóa các ngưỡng cắt (Tipping Points) tìm được từ phân tích kịch bản thành các biến số đơn giản (ví dụ: `Age_OK`).
*   **Nguyên lý Occam's Razor:** Ưu tiên mô hình đơn giản (Logistic Regression/Decision Tree) nếu sau khi tối ưu dữ liệu, chúng đạt hiệu năng tương đương với mô hình phức tạp.
