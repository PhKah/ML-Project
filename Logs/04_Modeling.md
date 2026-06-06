# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với Pure Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu hệ thống (Global Data Leakage).
    2. **[CHIẾN LƯỢC]** Di chuyển toàn bộ các bước tính toán thống kê (Scaling, Imputation) vào bên trong Pipeline để đảm bảo tính khách quan tuyệt đối.
    3. **[ĐÁNH GIÁ CÔNG BẰNG]** Sử dụng **F1-Score** làm metric chủ đạo để cân bằng giữa Precision và Recall.
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Phiên bản Triệt tiêu Rò rỉ & Tối ưu hóa F1.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (~160 đặc trưng **Dạng Thô**, bao gồm `pair_id`).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt và triết lý thiết kế sản phẩm:
1.  **Group-Aware Splitting:** Sử dụng `StratifiedGroupKFold` dựa trên `pair_id` để chia tập Train/Val/Test, đảm bảo cặp đôi A-B không bị xé lẻ, bảo vệ tính trung thực 100%.
2.  **Encapsulated Pure Pipeline (Strict Integrity):** Mọi bước biến đổi dữ liệu phải nằm trong `sklearn.pipeline.Pipeline`:
    *   `SimpleImputer(strategy='median')`: Điền khuyết cục bộ trong từng Fold của Cross-validation.
    *   `ColumnTransformer`: 
        *   **`MinMaxScaler`**: Dành cho nhóm `_gap` và các biến thang điểm (0-9, 1-10) để bảo toàn bản chất không âm và ý nghĩa vật lý của khoảng cách.
        *   **`StandardScaler`**: Dành cho nhóm `_surplus` và các biến liên tục (như `age`) để mô tả sự lệch pha so với kỳ vọng trung bình.
        *   **`OneHotEncoder`**: Chỉ áp dụng cho Logistic Regression để xử lý các biến định danh khách quan. 
    *   **LOẠI BỎ SMOTE:** Qua SMOTE Audit, chúng tôi phát hiện 22% dữ liệu sinh ra là "hạng mục ma" (race=1.5). Việc loại bỏ SMOTE giúp triệt tiêu hoàn toàn nguồn nhiễu này.
    *   **Class Weighting:** Sử dụng tham số nội tại của thuật toán để xử lý mất cân bằng lớp một cách tự nhiên.
3.  **Feature Selection (Noise Reduction):** Sàng lọc Top đặc trưng quan trọng nhất dựa trên mô hình Random Forest/XGBoost để loại bỏ các đặc trưng "nhiễu" từ quá trình FE bùng nổ (154 biến).
4.  **Balanced Optimization:** Sử dụng **F1-Score** làm metric chủ đạo trong suốt quá trình `GridSearchCV` và chọn Winner để đạt được sự cân bằng tối ưu giữa độ tin cậy và khả năng bao phủ.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 0: Phân loại đặc trưng** thành 3 nhóm (MinMax, Standard, Passthrough) bằng **Set Logic**.
- [x] **Bước 1: Chia tách dữ liệu Group Split** từ file dữ liệu thô.
- [x] **Bước 2: Xây dựng Pipeline phức hợp** tích hợp Imputer và Transformers song song.
- [x] **Bước 2b: Tích hợp One-Hot Encoding** (Chỉ cho Logistic Regression).
- [ ] **Bước 2d: Sàng lọc đặc trưng tinh hoa (Top Feature Selection)** để giảm nhiễu hệ thống.
- [ ] **Bước 3: GridSearch với Pure Pipeline sử dụng F1 Scorer**.
- [ ] **Bước 4: Threshold Shifting (Validation Set) dựa trên F1**.
- [ ] **Bước 5: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 9: Metric Realignment to F1
*   *Chiến lược: Đồng bộ hóa toàn bộ mục tiêu của hệ thống về phía F1-Score (Cân bằng Precision/Recall).*

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 7.1. Chân tướng của sự Rò rỉ Dữ liệu
*   Việc di chuyển Scaler vào Pipeline đã làm giảm điểm số thực tế từ ~0.38 (ảo) xuống ~0.34 (thật). Điểm ~0.34 hiện tại phản ánh chính xác năng lực dự báo trên dữ liệu hoàn toàn lạ.

### 🔍 7.2. Tại sao chọn F1 thay vì F0.5?
*   F1-score đảm bảo rằng hệ thống không chỉ "thận trọng" (Precision) mà còn phải "hiệu quả" (Recall) — bắt được nhiều cặp Match tiềm năng nhất có thể mà không làm loãng độ tin cậy quá mức.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Thiết kế Tầng Ứng dụng (Application Routing):** Mô hình hiện tại đã bao đóng (Encapsulated) toàn bộ tri thức về chuẩn hóa. Khi triển khai thực tế, tầng API chỉ cần nạp dữ liệu thô.

## 9. Bước tiếp theo
*   **Task 05: Evaluation & Final Reporting:** Chốt báo cáo với Dashboard chẩn đoán 5 tầng và chốt bộ đặc trưng quan trọng.
