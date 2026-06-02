# Task 04: Lựa chọn và Huấn luyện mô hình (Modeling)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Xây dựng, huấn luyện và tối ưu hóa các mô hình học máy để dự đoán khả năng `match`.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   Mô hình nào (Logistic Regression, Decision Tree, Random Forest, SVM) cho hiệu năng (F1-score) tốt nhất?
    *   Có hiện tượng Overfitting xảy ra khi sử dụng các mô hình phức tạp không?
    *   Các siêu tham số tối ưu cho từng mô hình là gì?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final.csv`.
*   **Mã nguồn:** `src/04_modeling.py`
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Bảng so sánh hiệu năng các mô hình trên tập Validation.

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Toàn bộ tập dữ liệu đã chuẩn hóa (23 thuộc tính).
*   **Phương pháp:** 
    *   **Chia dữ liệu (Hold-out):** Tỷ lệ Train (60%), Validation (20%), Test (20%).
    *   **Baseline:** Logistic Regression.
    *   **Model Selection:** Decision Tree, Random Forest, SVM.
    *   **Tuning:** GridSearchCV với 3-fold cross-validation trên tập Train.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Chia dữ liệu Train/Validation/Test**
- [x] **Bước 2: Thiết lập Baseline (Logistic Regression)**
- [x] **Bước 3: Huấn luyện và Tinh chỉnh các mô hình phức tạp**

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành.
*   **Mã nguồn:** `src/04_modeling.py`
*   **Chi tiết:** 
    *   Dữ liệu được chia tầng (Stratified) để giữ nguyên tỷ lệ lớp `match`.
    *   Thực hiện GridSearch cho DT, RF, và SVM.
    *   Kết quả thu được trên tập Validation (F1-score).

## 6. Kết quả & Kiểm chứng (Validation)
*   **Bảng so sánh hiệu năng (F1-score trên Validation set):**

| Mô hình | F1-score | Siêu tham số tốt nhất |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.2071 | Mặc định |
| **Decision Tree** | **0.3333** | `max_depth: 15`, `min_samples_split: 2` |
| **Random Forest** | 0.2880 | `max_depth: None`, `n_estimators: 100` |
| **SVM** | 0.2680 | `C: 10`, `kernel: 'rbf'` |

*   **Nhận xét:** 
    *   **Decision Tree** đạt kết quả cao nhất trên tập Validation.
    *   Các mô hình nhìn chung có F1-score thấp, phản ánh độ khó của bài toán khi dữ liệu rất mất cân bằng.
    *   Random Forest và SVM có thể cần dải siêu tham số rộng hơn hoặc kỹ thuật xử lý mất cân bằng (như SMOTE hoặc class_weight) để cải thiện thêm.

## 7. Khám phá quan trọng (Insights & Insights Update)
*   **Phát hiện 1:** Mô hình phi tuyến (Tree-based, SVM) vượt trội rõ rệt so với Logistic Regression tuyến tính.
*   **Phát hiện 2:** Decision Tree đơn giản lại cho kết quả tốt nhất, có thể do cấu trúc dữ liệu Dating có các quy tắc "cắt" rõ ràng (ví dụ: nếu `attr` > ngưỡng X thì khả năng match cao).

## 8. Bước tiếp theo
*   Chuyển sang Giai đoạn 5: Đánh giá mô hình cuối cùng trên tập Test và Rút ra tri thức (Feature Importance).
