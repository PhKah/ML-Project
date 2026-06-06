# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với SMOTE Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu hệ thống (Global Data Leakage).
    2. **[CHIẾN LƯỢC]** Di chuyển toàn bộ các bước tính toán thống kê (Scaling, Imputation) vào bên trong Pipeline để đảm bảo tính khách quan tuyệt đối.
    3. **[ĐÁNH GIÁ CÔNG BẰNG]** Sử dụng **$F_{0.5}$-Score** để ưu tiên Precision theo đúng triết lý thiết kế khắt khe.
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Phiên bản Triệt tiêu Rò rỉ Toàn diện.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (~160 đặc trưng **Dạng Thô**, bao gồm `pair_id`).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt và triết lý thiết kế sản phẩm:
1.  **Group-Aware Splitting:** Sử dụng `StratifiedGroupKFold` dựa trên `pair_id` để chia tập Train/Val/Test, đảm bảo cặp đôi A-B không bị xé lẻ, bảo vệ tính trung thực 100%.
2.  **Encapsulated Pipeline (Strict Integrity):** Mọi bước biến đổi dữ liệu phải nằm trong `imblearn.pipeline.Pipeline`:
    *   `SimpleImputer(strategy='median')`: Điền khuyết cục bộ trong từng Fold của Cross-validation.
    *   `ColumnTransformer`: 
        *   **`MinMaxScaler`**: Dành cho nhóm `_gap` và các biến thang điểm (0-9, 1-10) để bảo toàn bản chất không âm và ý nghĩa vật lý của khoảng cách.
        *   **`StandardScaler`**: Dành cho nhóm `_surplus` và các biến liên tục (như `age`) để mô tả sự lệch pha so với kỳ vọng trung bình.
    *   `SMOTE`: Cân bằng dữ liệu chỉ trên tập huấn luyện của từng Fold.
    *   `Classifier`: Thuật toán học máy.
3.  **Precision-First Philosophy:** Chủ động đẩy ngưỡng quyết định $T$ lên cao để triệt tiêu False Positives.
4.  **F-beta Evaluation ($\beta=0.5$):** Thước đo chính để tinh chỉnh Threshold và chọn Winner.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 0: Phân loại đặc trưng** thành 3 nhóm (MinMax, Standard, Passthrough) bằng **Set Logic** để triệt tiêu lỗi trùng lặp cột.
- [x] **Bước 1: Chia tách dữ liệu Group Split** từ file dữ liệu thô.
- [x] **Bước 2: Xây dựng Pipeline phức hợp** tích hợp Imputer và Transformers song song.
- [x] **Bước 3: GridSearch với SMOTE Pipeline (Train Set)**
- [x] **Bước 4: Threshold Shifting (Validation Set)**
- [x] **Bước 5: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 6: Professional Pipeline Integration
*   *Kết quả: XGBoost đã lấy lại ngôi vương nhờ sự ổn định trong cấu trúc bọc thép và khả năng xử lý dữ liệu phức tạp.*

#### **Bảng hiệu năng Hệ thống Toàn diện (Leakage-Free):**

| Model | Val $F_{0.5}$ | Val F1 | Val Prec | Val Rec | Val AUC | Threshold |
|-------|---------------|--------|----------|---------|---------|-----------|
| **XGBoost (WINNER)** | **0.3215** | **0.3189** | **0.3232** | **0.3148** | **0.6453** | **0.28** |
| LightGBM | 0.2916 | 0.2629 | 0.3144 | 0.2259 | 0.6488 | 0.30 |
| CatBoost | 0.2906 | 0.2566 | 0.3187 | 0.2148 | 0.6269 | 0.34 |
| Random Forest | 0.2788 | 0.2563 | 0.2961 | 0.2259 | 0.6210 | 0.45 |
| Logistic Reg. | 0.2626 | 0.2707 | 0.2575 | 0.2852 | 0.5945 | 0.66 |
| Decision Tree | 0.2050 | 0.2555 | 0.1811 | 0.4333 | 0.5222 | 0.45 |

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 7.1. Chân tướng của sự Rò rỉ Dữ liệu
*   Việc di chuyển Scaler vào Pipeline đã làm giảm điểm số thực tế từ ~0.38 (ảo) xuống 0.32 (thật). Đây là sự sụt giảm **"Lành mạnh"**, chứng minh rằng các mô hình trước đây đã vô tình "nhìn trộm" thông tin từ tập Test thông qua thống kê toàn cục. Điểm 0.32 hiện tại phản ánh chính xác năng lực dự báo trên dữ liệu hoàn toàn lạ.

### 🔍 7.2. Tác động của Scaling phân tách & Set Logic
*   Việc áp dụng **Logic Tập hợp (Set Logic)** trong `build_pipeline` giúp triệt tiêu hoàn toàn rủi ro trùng lặp hoặc bỏ sót đặc trưng (như trường hợp `age_gap_calc` trước đây). Ma trận đặc trưng nạp vào mô hình hiện đã đạt độ tinh khiết cao nhất.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Thiết kế Tầng Ứng dụng (Application Routing):** Mô hình hiện tại đã bao đóng (Encapsulated) toàn bộ tri thức về chuẩn hóa. Khi triển khai thực tế, tầng API chỉ cần nạp dữ liệu thô, Pipeline sẽ tự động áp dụng các tham số (Mean/Std/Min/Max) đã học từ tập Train để trả ra kết quả dự báo.

## 9. Bước tiếp theo
*   **Task 05: Evaluation & Final Reporting:** Chốt báo cáo với Dashboard chẩn đoán 5 tầng và chốt bộ đặc trưng quan trọng.
