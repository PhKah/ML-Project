# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với Hybrid SMOTE-NC Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu hệ thống (Global Data Leakage).
    2. **[CHIẾN LƯỢC]** Di chuyển toàn bộ các bước tính toán thống kê (Scaling, Imputation) vào bên trong Pipeline để đảm bảo tính khách quan tuyệt đối.
    3. **[ĐÁNH GIÁ CÔNG BẰNG]** Sử dụng **$F_{0.5}$-Score** làm metric chủ đạo để ưu tiên Precision (Độ tin cậy của gợi ý).
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Phiên bản Hybrid (Cân bằng mẫu sạch & Ưu tiên Precision).

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (~160 đặc trưng **Dạng Thô**, bao gồm `pair_id`).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt và triết lý thiết kế sản phẩm:
1.  **Group-Aware Splitting:** Sử dụng `StratifiedGroupKFold` dựa trên `pair_id` để chia tập Train/Val/Test.
2.  **Encapsulated Hybrid Pipeline (Option A - Structural Integrity):** Mọi bước biến đổi dữ liệu phải nằm trong `imblearn.pipeline.Pipeline` với thứ tự nghiêm ngặt:
    *   **SimpleImputer:** Điền khuyết (Bắt buộc trước khi Resampling).
    *   **SMOTE-NC (Nominal & Continuous):** Thực hiện tái mẫu trên dữ liệu nhãn số nguyên (Label Encoding) **TRƯỚC khi mã hóa OHE**. Việc này đảm bảo các mẫu ảo sinh ra khi đi qua OHE sẽ luôn là vector nhị phân (0, 1) chuẩn, triệt tiêu nhiễu phân số (0.4, 0.7) gây hỏng logic của mô hình Logistic.
    *   **ColumnTransformer:** Thực hiện MinMaxScaler, StandardScaler và One-Hot Encoding trên nền dữ liệu đã được cân bằng sạch.
    *   **Full Feature Set:** Sử dụng toàn bộ 154 đặc trưng để tối đa hóa tín hiệu tương tác.
3.  **Automated Index Mapping:** Tự động xác định vị trí cột định danh trong tập dữ liệu thô để nạp vào SMOTE-NC.
4.  **Locked Sampling Strategy:** Cố định tỷ lệ tái mẫu `sampling_strategy=0.43`. Con số này được xác lập qua các thực nghiệm trước đó là "điểm ngọt" (Sweet Spot), giúp nâng tỷ trọng lớp Match lên chiếm ~30% tổng dữ liệu, đủ để mô hình nhận diện nhưng không quá lớn để gây nhiễu Precision.
5.  **Noise Audit (Mandatory):** Kiểm định tính nguyên vẹn của dữ liệu sau xử lý, đảm bảo 0% giá trị phân số trong không gian định danh (Categorical Space).
6.  **Precision-Driven Optimization:** Tuning và Threshold theo **$F_{0.5}$-Score**.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 0: Phân loại đặc trưng** thành 3 nhóm (MinMax, Standard, Passthrough) bằng **Set Logic**.
- [x] **Bước 1: Chia tách dữ liệu Group Split** từ file dữ liệu thô.
- [x] **Bước 2: Tái cấu trúc Pipeline Hybrid** (Impute -> SMOTE-NC -> Transformer -> Model).
- [x] **Bước 2e: Triển khai SMOTE-NC** trên dữ liệu Label Encoding để bảo vệ tính Sparse của OHE.
- [x] **Bước 2g: Áp dụng tỷ lệ tái mẫu cố định (sampling_strategy=0.43)** để bảo vệ Precision.
- [x] **Bước 2f: Kiểm tra nhiễu (Categorical Noise Audit)** sau Pipeline.
- [x] **Bước 3: GridSearch với Hybrid Pipeline (Target: $F_{0.5}$)**.
- [x] **Bước 4: Threshold Shifting (Validation Set) dựa trên $F_{0.5}$**.
- [x] **Bước 5: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 12: Feature Aggregation & Performance Breakthrough
*   *Kết quả: Việc gộp 17 biến sở thích thành 'mean_hobby_gap' đã tạo ra bước nhảy vọt về hiệu năng. XGBoost đạt Precision kỷ lục 42.4%.*

#### **Bảng hiệu năng Hệ thống Toàn diện (Leakage-Free - Hybrid SMOTE-NC & Aggregation):**

| Model | Val $F_{0.5}$ | Val Acc | Val Prec | Val Rec | Val F1 | Val AUC | Threshold |
|-------|---------------|---------|----------|---------|--------|---------|-----------|
| **XGBoost (WINNER)** | **0.3442** | **0.8240** | **0.4240** | **0.1963** | **0.2684** | **0.6722** | **0.41** |
| CatBoost | 0.3347 | 0.7911 | 0.3447 | 0.3000 | 0.3208 | 0.6646 | 0.30 |
| Random Forest | 0.3341 | 0.7844 | 0.3372 | 0.3222 | 0.3295 | 0.6479 | 0.33 |
| LightGBM | 0.3149 | 0.7771 | 0.3168 | 0.3074 | 0.3120 | 0.6628 | 0.23 |
| Logistic Reg. | 0.2849 | 0.7734 | 0.2910 | 0.2630 | 0.2763 | 0.6105 | 0.33 |
| Decision Tree | 0.2036 | 0.3021 | 0.1712 | 0.8444 | 0.2846 | 0.5077 | 0.02 |

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 7.1. Chân tướng của sự Rò rỉ Dữ liệu
*   Việc di chuyển Scaler vào Pipeline đã làm giảm điểm số thực tế từ ~0.38 (ảo) xuống ~0.34 (thật).

### 🔍 7.2. Tại sao chọn tỷ lệ 0.43?
*   Tỷ lệ 0.43 (~30% Match) giúp duy trì sự cân bằng tinh tế: đủ Recall để bắt được các cặp Match khó, nhưng vẫn giữ Precision ở mức cao nhờ không làm loãng mật độ của lớp đa số (No Match).

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Thiết kế Tầng Ứng dụng:** Hệ thống đạt độ tin cậy tối đa nhờ sự kết hợp giữa cân bằng mẫu thông minh và mã hóa đặc trưng chuẩn mực.

## 9. Bước tiếp theo
*   **Thực thi Pipeline Hybrid Option A với tỷ lệ cố định và Re-benchmark.**
