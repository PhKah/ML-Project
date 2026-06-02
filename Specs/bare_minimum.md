# Bare Minimum Specification

## 1. Mục tiêu cốt lõi
*   **Bài toán:** Phân loại nhị phân (Binary Classification).
*   **Target Variable:** `match` (1: Cả hai cùng đồng ý, 0: Ngược lại).
*   **Metric đánh giá chính:** Accuracy, Confusion Matrix, Precision, Recall, F1-score (đặc biệt quan trọng do mất cân bằng lớp) và ROC-AUC.
*   **Biến phụ trợ:** `dec` (Quyết định cá nhân của chủ thể).

## 2. Dữ liệu đầu vào "Sống còn" (Features)
Để mô hình chạy được mức tối thiểu, cần tập trung vào các nhóm này:
*   **Cá nhân (Subject):** `gender`, `age`, `race`, `imprace` (tầm quan trọng của sắc tộc), `imprelig` (tầm quan trọng của tôn giáo).
*   **Sở thích (Preferences - Time 1):** `attr1_1` (Hấp dẫn), `sinc1_1` (Chân thành), `intel1_1` (Thông minh), `fun1_1` (Vui vẻ), `amb1_1` (Tham vọng), `shar1_1` (Sở thích chung).
*   **Đánh giá đối phương (Partner Scorecard):** `attr`, `sinc`, `intel`, `fun`, `amb`, `shar` (Điểm số đối phương chấm cho mình sau cuộc gặp).
*   **Biến giả thuyết:** `condtn` (1: Dưới 10 lựa chọn, 2: Trên 20 lựa chọn).

## 3. Giả thuyết nghiên cứu (Research Questions)
*   **Nghịch lý lựa chọn:** Liệu tỷ lệ `match` ở `condtn=2` có thấp hơn đáng kể so với `condtn=1` không?
*   **Yếu tố quyết định:** Thuộc tính nào (ví dụ: `attr` vs `intel`) có trọng số cao nhất trong việc dự đoán `match`?
