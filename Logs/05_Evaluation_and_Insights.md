# Task 05: Đánh giá mô hình & Rút ra tri thức (Evaluation & Insights)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Thực hiện đánh giá khách quan cuối cùng trên tập Test và phân tích các đặc trưng quan trọng nhất để rút ra Insight cho bài toán.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation & Insights) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   Hiệu năng thực tế của mô hình trên dữ liệu chưa từng thấy là bao nhiêu?
    *   Yếu tố nào (Hấp dẫn, Vui vẻ, hay Thông minh) thực sự quyết định khả năng match?
    *   Kết quả mô hình có củng cố cho giả thuyết "Nghịch lý lựa chọn" không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final.csv`.
*   **Mã nguồn:** `src/05_evaluation.py`
*   **Đầu ra:** 
    *   File log này (`Logs/05_Evaluation_and_Insights.md`).
    *   Biểu đồ Ma trận nhầm lẫn (`plots/05_confusion_matrix.png`).
    *   Biểu đồ Đường cong ROC (`plots/05_roc_curve.png`).
    *   Biểu đồ Tầm quan trọng của thuộc tính (`plots/05_feature_importance.png`).

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Toàn bộ tập dữ liệu đã chuẩn hóa, tập trung vào tập Test (20%).
*   **Phương pháp:** 
    *   **Đánh giá cuối cùng:** Sử dụng mô hình Decision Tree (`max_depth=15`) dự báo trên tập Test.
    *   **Tính toán Metric:** Accuracy, Confusion Matrix, Precision, Recall, F1-score và AUC.
    *   **Phân tích đặc trưng:** Trích xuất `feature_importances_`.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Dự báo và Đánh giá trên tập Test**
- [x] **Bước 2: Trực quan hóa kết quả**
- [x] **Bước 3: Phân tích Feature Importance**

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành.
*   **Mã nguồn:** `src/05_evaluation.py`
*   **Chi tiết:** 
    *   Sử dụng môi trường Conda `course` với đầy đủ thư viện `matplotlib`, `seaborn`, `sklearn`.
    *   Mô hình được huấn luyện lại trên tập Train và đánh giá trên 1676 bản ghi tập Test.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Chỉ số hiệu năng (Test Set):**
    *   **Accuracy:** **0.77** (Khá cao, nhưng do lớp 0 chiếm đa số).
    *   **F1-score (lớp 1):** **0.31** (Dự đoán đúng khoảng 1/3 số cặp thực sự match).
    *   **ROC-AUC:** Phản ánh khả năng phân loại khá tốt giữa hai nhóm.
*   **Ma trận nhầm lẫn:** Mô hình dự đoán đúng 87% các trường hợp không match, nhưng vẫn gặp khó khăn trong việc bắt được toàn bộ các cặp match (Recall lớp 1 là 31%).

## 7. Khám phá quan trọng (Insights & Insights Update)
*   **Phát hiện 1 (Top Features):** 
    1.  **`attr` (Hấp dẫn - Dynamic):** Đứng đầu với 12%. Điều này cho thấy ngoại hình vẫn là yếu tố "mở cửa" quan trọng nhất.
    2.  **`age_diff` (Chênh lệch tuổi):** Đứng thứ hai (~8.4%). Càng gần tuổi nhau, xác suất match càng cao.
    3.  **`amb` (Tham vọng) & `shar` (Sở thích chung):** Đóng góp quan trọng vào quyết định cuối cùng.
*   **Phát hiện 2 (Static vs Dynamic):** Các biến sở thích ban đầu (`attr1_1`, `fun1_1`) vẫn lọt vào top 10 nhưng có trọng số thấp hơn hẳn các biến chấm điểm thực tế sau cuộc gặp.
*   **Kết luận giả thuyết:** 
    *   **Nghịch lý lựa chọn:** Được củng cố bởi EDA ở Task 02.
    *   **Yếu tố quyết định:** `attr` là biến quan trọng nhất trong mô hình Decision Tree, nhưng `age_diff` là một phát hiện bất ngờ có giá trị dự báo cao.

## 8. Bước tiếp theo
*   Tổng hợp toàn bộ nội dung vào Báo cáo đồ án (Report).
*   Đóng gói mã nguồn và dữ liệu.
