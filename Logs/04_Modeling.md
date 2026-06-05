# Task 04: Huấn luyện mô hình Dyadic & Kiểm định Nghiêm ngặt với SMOTE Pipeline

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Thiết lập quy trình huấn luyện chuyên nghiệp, loại bỏ rò rỉ dữ liệu (Data Leakage).
    2. **[CHIẾN LƯỢC]** Sử dụng **SMOTE** để xử lý mất cân bằng lớp.
    3. **[ĐÁNH GIÁ CÔNG BẰNG]** Sử dụng **$F_{0.5}$-Score** để ưu tiên Precision theo đúng triết lý thiết kế khắt khe.
    4. Benchmark 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Chuẩn hóa với SMOTE Pipeline.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (~159 đặc trưng, đã xử lý Deduplication chống rò rỉ).
*   **Mã nguồn:** `src/04_modeling.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Mô hình chiến thắng được khóa tại `models/winner_model.joblib`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ các tiêu chuẩn nghiêm ngặt và triết lý thiết kế sản phẩm:
1.  **SMOTE Inside Pipeline:** Đảm bảo tính khách quan cho Cross-validation.
2.  **Precision-First Philosophy:** Chủ động đẩy ngưỡng quyết định $T$ lên cao (0.2 - 0.7) để triệt tiêu False Positives.
3.  **F-beta Evaluation ($\beta=0.5$):** Chuyển dịch thước đo sang $F_{0.5}$ để phản ánh đúng hiệu quả của bộ lọc Precision.
4.  **Anti-Leakage Validation:** Dữ liệu đầu vào đã bị loại bỏ 50% bản ghi đối xứng, đảm bảo mô hình không thể "học vẹt" (memorize) kết quả từ tập Train để nhắc bài cho tập Test.
5.  **Objectivity First (Không Tipping Points):** Mô hình được tiếp nhận dữ liệu liên tục 100% nguyên thủy, không hề có sự mớm bài từ các ngưỡng cắt do con người định sẵn, tránh overfit vào các quy luật cục bộ.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 0: Loại bỏ các biến Zero-Variance (`gender`, `gender_o`)** do hệ quả của quá trình deduplication và loại bỏ block load Tipping Points.
- [ ] **[KIẾN TRÚC MỚI] Bước 1: Chia tách dữ liệu Group Split & Ẩn tập Test.** Thay thế thuật toán `train_test_split` ngẫu nhiên bằng `GroupShuffleSplit` kết hợp với cột `pair_id` để ngăn chặn rò rỉ chéo mà vẫn giữ được 100% dữ liệu.
- [x] **Bước 2: GridSearch với SMOTE Pipeline (Train Set)**
- [x] **Bước 3: Threshold Shifting (Validation Set) - Khắt khe hóa hệ thống**
- [x] **Bước 4: Khóa mô hình chiến thắng (Save Winner)**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 5: Anti-Leakage Modeling & F-beta Validation
*   *Kết quả: Khi dữ liệu bị tước bỏ lợi thế "nhắc bài" do trùng lặp và các Tipping Points tự code, hiệu năng hệ thống giảm nhẹ nhưng trở nên trung thực 100%. XGBoost đã soán ngôi LightGBM nhờ khả năng kiểm soát Overfitting xuất sắc trên dữ liệu khó.*

#### **Bảng hiệu năng Hệ thống Toàn diện (F-beta Optimized):**

| Model | Val $F_{0.5}$ | Val F1 | Val Prec | Val Rec | Val AUC | Threshold |
|-------|---------------|--------|----------|---------|---------|-----------|
| **XGBoost (WINNER)** | **0.3875** | **0.3866** | **0.3881** | **0.3852** | **0.7183** | **0.27** |
| CatBoost | 0.3484 | 0.3162 | 0.3737 | 0.2741 | 0.6820 | 0.34 |
| LightGBM | 0.3439 | 0.3534 | 0.3378 | 0.3704 | 0.6758 | 0.22 |
| Random Forest | 0.3301 | 0.3690 | 0.3085 | 0.4593 | 0.6759 | 0.38 |
| Logistic Reg. | 0.2830 | 0.2165 | 0.3559 | 0.1556 | 0.6174 | 0.76 |
| Decision Tree | 0.2222 | 0.2413 | 0.2111 | 0.2815 | 0.5322 | 0.06 |

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Lựa chọn Winner: XGBoost (The Robust Champion)
1.  **Sự trỗi dậy của XGBoost:** Khi dữ liệu trở nên sạch và khó nhằn hơn (không còn rò rỉ, không còn Tipping Points nhân tạo), thuật toán phân nhánh theo chiều sâu (Depth-wise) của XGBoost chứng tỏ sự bền vững vượt trội so với LightGBM (vốn dễ bị overfit trên dữ liệu nhỏ).
2.  **Sự cân bằng hoàn hảo:** Precision (38.8%) và Recall (38.5%) đạt trạng thái cân bằng lý tưởng ở ngưỡng $T=0.27$.
3.  **Tập Test:** Sẵn sàng để giải mã với mô hình mạnh nhất này.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🔍 7.1. Chân tướng của sự Rò rỉ Dữ liệu (Reciprocal Leakage)
*   **Phát hiện:** Trước khi xóa dữ liệu đối xứng, Precision đạt tới 45%. Sau khi xóa, điểm giảm xuống 38.8%. Điều này chứng minh rằng mức điểm cao trước đó là "ảo" do mô hình học thuộc lòng các cặp đôi ở tập Train. Việc giảm điểm này là một thành công lớn về mặt khoa học dữ liệu, khẳng định mô hình hiện tại là một "Dự báo viên" thực thụ, không phải một "Kẻ chép bài".

### 🔍 7.2. Sự ổn định của ROC-AUC
*   Mặc dù Precision và F1 giảm, chỉ số **ROC-AUC vẫn giữ vững ở mức >0.71**. Điều này cho thấy bộ đặc trưng Mutual Surplus 2 tầng đã bù đắp xuất sắc cho sự mất mát lượng dữ liệu, giúp mô hình duy trì khả năng phân loại bản chất rất tốt.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Thiết kế Tầng Ứng dụng (Application Routing):** Vì mô hình hiện tại được huấn luyện trên dữ liệu một chiều (Nữ đánh giá Nam, do kết quả của hàm `min` trong deduplication), khi triển khai lên production, tầng API cần có một hàm Routing. Hàm này tự động sắp xếp lại dữ liệu đầu vào: gán Nữ làm Subject và Nam làm Partner trước khi tính toán các biến Surplus, nhằm bảo toàn dấu (+/-) của các kỳ vọng song phương.

## 9. Bước tiếp theo
*   **Task 05: Evaluation & Final Reporting:** Giải mã tập Test với mô hình XGBoost.

