# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với SMOTE Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu (Data Leakage) khi sử dụng các kỹ thuật Resampling.
    2. **[CHIẾN LƯỢC]** Sử dụng **SMOTE** để xử lý mất cân bằng lớp một cách hệ thống bên trong Pipeline.
    3. **[KIỂM ĐỊNH]** Phân định rõ vai trò của tập dữ liệu: Tối ưu trên tập Train/Val, **Báo cáo trung thực trên tập Test**.
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Chuẩn hóa với SMOTE Pipeline.
*   **Giả thuyết/Câu hỏi:** 
    * Việc cung cấp dữ liệu ở dạng thô (Granular Hobbies) có giúp mô hình tìm thấy những tín hiệu tinh vi hơn không?
    * Mô hình nào xử lý tốt nhất khi số lượng đặc trưng tăng lên 119 cột?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (119 đặc trưng Granular & Objective).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Bảng kết quả benchmark (Validation only).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt:
1.  **SMOTE Inside Pipeline:** Đảm bảo tính khách quan cho Cross-validation.
2.  **Granular Input:** Đưa toàn bộ 17 sở thích rời rạc của cả hai bên vào mô hình.
3.  **Threshold Optimization:** Tìm ngưỡng $T$ tối ưu trên tập Validation.
4.  **Test Set Isolation:** Tập Test được lưu riêng và **ẩn hoàn toàn** trong giai đoạn này để tránh rò rỉ thông tin vào quá trình chọn mô hình.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 1: Chia tách dữ liệu & Ẩn tập Test**
- [x] **Bước 2: GridSearch với SMOTE Pipeline (Train Set)**
- [x] **Bước 3: Threshold Tuning (Validation Set)**
- [x] **Bước 4: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 5: Granular Dyadic Benchmark & Model Locking
*   *Kết quả: **LightGBM** xuất sắc dẫn đầu với điểm Val F1 vượt ngưỡng 0.4. Mô hình đã được khóa để đánh giá cuối cùng.*

#### **Bảng so sánh hiệu năng trên tập VALIDATION:**

| Model | Val F1 | Val Prec | Val Rec | Val AUC | Threshold |
|-------|--------|----------|---------|---------|-----------|
| **LightGBM** | **0.4134** | 0.3175 | 0.5926 | **0.7164** | 0.17 |
| CatBoost | 0.3837 | **0.3420** | 0.4370 | 0.6929 | 0.28 |
| Random Forest | 0.3815 | 0.2744 | 0.6259 | 0.6933 | 0.35 |
| XGBoost | 0.3720 | 0.2575 | **0.6704** | 0.6947 | 0.14 |
| Logistic Reg. | 0.3256 | 0.2100 | 0.7296 | 0.6163 | 0.39 |
| Decision Tree | 0.2885 | 0.1777 | 0.7667 | 0.5493 | 0.23 |

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Lựa chọn Winner: LightGBM
1.  **Hiệu năng vượt trội:** Val F1 đạt 0.41, cao nhất trong các mô hình thử nghiệm.
2.  **Cân bằng tốt:** Sự kết hợp giữa Precision (0.32) và Recall (0.59) ở ngưỡng $T=0.17$ mang lại kết quả ổn định.
3.  **Tập Test:** Hiện đang được lưu trữ tại `Data/test_set_hidden.csv` và sẽ chỉ được mở ra ở Giai đoạn 5.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 Top 10 Đặc trưng quan trọng (LightGBM)

| Rank | Feature | Importance | Ý nghĩa Insight |
|------|---------|------------|-----------------|
| 1 | `age_gap_calc` | 110 | **Khoảng cách tuổi:** Yếu tố quyết định số 1. |
| 2 | `int_corr` | 76 | Tương quan sở thích cá nhân. |
| 3 | `tvsports_o` | 60 | **Granular Hobby:** Sở thích xem thể thao của đối phương có tín hiệu dự báo mạnh. |
| 4 | `shar1_1` | 50 | Subject tìm kiếm sở thích chung. |
| 5 | `shar1_1_o` | 48 | Partner tìm kiếm sở thích chung. |

## 9. Bước tiếp theo
*   **Task 05: Evaluation & Final Reporting:** Giải mã tập Test và thực hiện chẩn đoán 5 cấp độ cho mô hình LightGBM.
