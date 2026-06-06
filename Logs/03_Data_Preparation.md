# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Chuyển đổi sang quy trình **Trích xuất Dữ liệu Thô Tinh khiết** (Raw Feature Extraction) nhằm triệt tiêu hoàn toàn rò rỉ dữ liệu hệ thống (Global Data Leakage).
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) - Phiên bản Chuẩn hóa Kiến trúc.
*   **Chiến lược Tương hợp Đa tầng (Multi-level Compatibility Distillation):** 
    1.  **Hobby Similarity:** Tính toán khoảng cách sở thích (`abs diff`) thô.
    2.  **Mutual Surplus (2 Tầng):** Tính toán thặng dư kỳ vọng (phép trừ có dấu) trên lăng kính 3_1 và 5_1 ở dạng số đo nguyên bản.
    3.  **Anti-Leakage (Strict Isolation):** Loại bỏ hoàn toàn các bước tính toán thống kê toàn cục (Mean, Std, Median) ở pha này. Mọi phép chuẩn hóa sẽ được thực hiện bên trong Pipeline huấn luyện để đảm bảo không rò rỉ thông tin từ tập Test vào tập Train.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`) chứa ~160 đặc trưng **Dạng Thô** (chưa scale, chưa điền khuyết muộn).

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Cách ly Thống kê"** (Statistical Isolation):
1.  **Pure Data Extraction:** Chỉ thực hiện các biến đổi mang tính logic (Join, Diff, Surplus) trên từng dòng dữ liệu.
2.  **No Global Imputation:** Kiên quyết không điền khuyết bằng Trung vị toàn cục (Global Median) tại đây để tránh rò rỉ thông tin từ tập Test vào tập Train.
3.  **No Pre-Scaling:** Loại bỏ Scaler. Việc lựa chọn `MinMaxScaler` cho Gaps (giữ tính không âm và dải vật lý 0-9) và `StandardScaler` cho Surplus (biến thiên âm-dương) sẽ được thực hiện trong `src/04`.
4.  **Referential Integrity:** Duy trì việc loại bỏ "Bóng ma" để bảo vệ độ sạch của hồ sơ thực thể.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 5: Pipeline Overhaul - Leakage Eradication (Đã hoàn thành)
- [x] **[DELETION]** Gỡ bỏ hàm `apply_scaling` và các bộ biến đổi tỷ lệ toàn cục.
- [x] **[DELETION]** Gỡ bỏ lệnh `fillna(median)` toàn cục cho tập Interaction.
- [x] Duy trì bộ 24 biến Mutual Surplus (S và P) ở dạng thô.
- [x] Duy trì cột `pair_id` cho Group Split.
- [x] Đảm bảo logic xóa Bóng ma và tính toàn vẹn tham chiếu.

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 4: Ultimate Cognitive Data Prep
*   *Mục tiêu: Bắt trọn mọi sắc thái tương hợp.*

### ✅ Hoàn thành Phase 5: Raw Data Standardization
*   *Mục tiêu: Đưa Pipeline về chuẩn khoa học dữ liệu, biến Giai đoạn 3 thành 'Data Factory' cung cấp nguyên liệu thô sạch.*

#### **Bước A: Thanh lọc thực thể (Entity Cleaning)**
- **A1:** Xóa 7 bóng ma thiếu thông tin hồ sơ.
- **A2:** Clip Outliers để bảo vệ phân phối gốc.

#### **Bước B: Hợp nhất và Trích xuất (Merge & Extract)**
- **B1:** Join hồ sơ Dyadic.
- **B2:** Tính toán 17 Hobby Gaps và 24 Mutual Surplus (Thô).
- **B3:** Xuất `pair_id` và giữ nguyên toàn bộ 8,210 dòng tương tác.

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu đầu ra | 8,210 hàng × 157 cột |
| **Trạng thái Scaling** | **Pure Raw (Chưa chuẩn hóa)** |
| **Trạng thái Imputation** | **Partial (Chỉ ở cấp User)** |
| Rủi ro rò rỉ toàn cục | **0% (Đã triệt tiêu)** ✓ |

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Triết lý Chống rò rỉ Hệ thống (Global Leakage Prevention):** Việc tính toán Mean/Std hay Median trên toàn bộ 8,210 dòng trước khi chia tập Test là một lỗi chí mạng (Data Snooping). Thông tin từ tập Test sẽ bị "thẩm thấu" vào các tham số chuẩn hóa. Bằng cách trích xuất dữ liệu ở dạng **Thô Tinh khiết**, chúng ta thiết lập một "bức tường lửa" bảo vệ tính khách quan tuyệt đối cho mô hình.
*   **Bảo toàn phân phối gốc:** Giữ nguyên giá trị thô giúp mô hình ở Giai đoạn 4 có cái nhìn toàn cảnh và chính xác nhất về phân phối tự nhiên của dữ liệu trước khi áp dụng các bộ biến đổi (Scalers) phù hợp.

## 9. Bước tiếp theo
*   Cập nhật `src/04_modeling.py` để tích hợp `ColumnTransformer` phức hợp (MinMax + Standard) và `SimpleImputer` vào Pipeline huấn luyện.
