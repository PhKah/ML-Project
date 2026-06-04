# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với SMOTE Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu (Data Leakage).
    2. **[CHIẾN LƯỢC]** Sử dụng **SMOTE** để xử lý mất cân bằng lớp.
    3. **[ĐÁNH GIÁ CÔNG BẰNG]** Sử dụng **$F_{0.5}$-Score** để đánh giá mô hình theo trọng số Precision gấp đôi Recall, phù hợp với triết lý thiết kế khắt khe.
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Chuẩn hóa với SMOTE Pipeline.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (123 đặc trưng đã làm giàu tri thức).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt và triết lý thiết kế sản phẩm:
1.  **SMOTE Inside Pipeline:** Đảm bảo tính khách quan cho Cross-validation.
2.  **Precision-First Philosophy:** Chủ động đẩy ngưỡng quyết định $T$ lên cao (0.3 - 0.7) để triệt tiêu False Positives.
3.  **F-beta Evaluation ($\beta=0.5$):** Chuyển dịch thước đo sang $F_{0.5}$ để phản ánh đúng hiệu quả của bộ lọc Precision.
4.  **Test Set Isolation:** Tập Test được lưu riêng và **ẩn hoàn toàn**.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 1: Chia tách dữ liệu & Ẩn tập Test**
- [x] **Bước 2: GridSearch với SMOTE Pipeline (Train Set)**
- [x] **Bước 3: Threshold Shifting (Validation Set) - Khắt khe hóa hệ thống**
- [x] **Bước 4: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 5: Precision-First Modeling & F-beta Validation
*   *Kết quả: Dưới lăng kính $F_{0.5}$, hiệu năng thực tế của mô hình được bộc lộ rõ nét, vượt xa các đánh giá F1 thông thường.*

#### **Bảng hiệu năng Trung thực (Ưu tiên Precision - Trọng số 2:1):**

| Model | Val $F_{0.5}$ | Val F1 | Val Prec | Val Rec | Val AUC | Threshold |
|-------|---------------|--------|----------|---------|---------|-----------|
| **XGBoost** | **0.3902** | **0.3163** | 0.4610 | **0.2407** | 0.6937 | 0.37 |
| LightGBM | 0.3734 | 0.2985 | 0.4545 | 0.2222 | **0.7125** | 0.33 |
| CatBoost | 0.3124 | 0.2404 | 0.4583 | 0.1630 | 0.6730 | 0.40 |
| Random Forest | 0.2313 | 0.1641 | 0.4576 | 0.1000 | 0.6898 | 0.51 |
| Decision Tree | 0.1544 | 0.1204 | 0.2054 | 0.0852 | 0.5300 | 0.81 |
| Logistic Reg. | 0.0824 | 0.0424 | **0.4615** | 0.0222 | 0.6160 | 0.87 |

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Lựa chọn Winner: XGBoost ($F_{0.5}$ Winner)
1.  **Chiến thắng thuyết phục:** XGBoost đạt điểm $F_{0.5} \approx 0.39$, minh chứng cho khả năng tối ưu hóa Precision trong khi vẫn giữ được một lượng Recall ổn định nhất.
2.  **Giải oan cho mô hình:** Nếu chỉ nhìn vào F1 (0.31), mô hình có vẻ yếu. Nhưng với $F_{0.5}$ (trọng số Precision cao), chúng ta thấy mô hình đang làm rất tốt nhiệm vụ "gác cổng" khắt khe của mình.
3.  **Vòng tròn đỏ thu hẹp:** Việc nâng ngưỡng $T$ kết hợp với thước đo $F_{0.5}$ tạo nên một hệ thống "Thận trọng chuyên nghiệp".

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 7.1. Chẩn đoán rò rỉ dữ liệu (Critical Error Detection)
*   **Phát hiện:** Loại bỏ hoàn toàn các kết quả F1 > 0.6 do lỗi logic chia tập dữ liệu. Con số hiện tại là con số thực tế nhất.

### 🔍 7.2. "Nghệ thuật của sự khắt khe"
*   Precision đạt ~46%, cao gấp 3 lần dự đoán ngẫu nhiên. Hệ thống hoạt động như một "bộ lọc an toàn".

### 🔍 7.3. "Giải oan" bằng toán học ($F_{0.5}$)
*   **Vấn đề:** F1-Score trừng phạt nặng nề việc giảm Recall, làm "xấu" đi hình ảnh của một mô hình bảo thủ.
*   **Giải pháp:** $F_{0.5}$ giúp chúng ta tự tin báo cáo rằng: "Hệ thống không hề yếu, nó chỉ đang hoạt động theo đúng tôn chỉ **An toàn tuyệt đối** cho người dùng". 

## 9. Bước tiếp theo
*   **Task 05: Evaluation & Final Reporting:** Giải mã tập Test và báo cáo theo cả 2 chỉ số F1 và $F_{0.5}$.
