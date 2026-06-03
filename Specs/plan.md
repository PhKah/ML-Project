# Project Implementation Framework (Framework-First Approach)

Kế hoạch này tuân thủ nguyên tắc **"Dữ liệu quyết định Công cụ"**, bao gồm 5 giai đoạn cốt lõi của vòng đời Khoa học Dữ liệu, bám sát giáo trình môn học.

## Giai đoạn 1: Định hình bài toán (Framing & Approach)
*   **Hiểu rõ nhu cầu thực tế (Business understanding):** Dự đoán khả năng "Match" giữa hai người trong sự kiện Speed Dating.
*   **Đưa về bài toán Học máy (Analytic approach):**
    *   **Loại bài toán:** Học có giám sát (Supervised Learning) - **Phân loại nhị phân (Binary Classification)**.
    *   **Không gian bài toán:** Xác định tập đặc trưng đầu vào ($X$) và nhãn đầu ra ($y$ - `match`).

## Giai đoạn 2: Xử lý Dữ liệu thô (Data Acquisition & Understanding)
*   **Thu thập (Acquisition):** Sử dụng tập dữ liệu Speed Dating (lưu tại `Data/`).
*   **Khám phá và Thấu hiểu (Data Understanding - EDA):** 
    *   Sử dụng thống kê mô tả và trực quan hóa (biểu đồ Histogram, Boxplot, Scatter plot).
    *   Phát hiện các điểm bất thường (anomalies), giá trị khuyết thiếu (missing values), và sự mất cân bằng lớp (class imbalance).

## Giai đoạn 3: Chuẩn bị và Tiền xử lý dữ liệu (Data Preparation)

### Chuẩn bị tuân thủ 6 nguyên lý tiền xử lý (từ Reflection):

1. **Entity vs Relationship Cleaning** (KHÔNG phải one-size-fits-all):
   *   **Dữ liệu Thực thể (Users/Entities):** Ưu tiên IMPUTATION (điền median/mode) để bảo toàn hồ sơ.
   *   **Dữ liệu Quan hệ (Interactions/Relationships):** Ưu tiên DELETION nếu quá thiếu (> 50% một block) để giữ tính xác thực.

2. **Order-Dependent Feature Engineering** (DAG - tuân thứ tự):
   *   Tính biến nguồn TRƯỚC khi scale (ví dụ: age_map trước scaled age)
   *   Xóa entity TRƯỚC khi đồng bộ relationship
   *   Vẽ DAG của dependencies nếu FE phức tạp

3. **Scaling Strategy** (Không phải mọi cột đều cần một cách):
   *   **MinMax [0,1]:** Cho dữ liệu có thang đo rõ (1-10 scales như hobbies, ratings)
   *   **StandardScaler (Z-score):** Cho dữ liệu liên tục (age, income, correlations)
   *   Lựa chọn dựa trên bản chất dữ liệu, không phải quy tắc chung

4. **Meaningful Aggregation** (Thay vì PCA máy móc):
   *   Gộp biến dựa trên correlation heatmap hoặc domain knowledge
   *   Tính mean/weighted average cho mỗi nhóm có ý nghĩa
   *   Lợi ích: Giảm chiều + Tăng khả năng giải thích + Tăng robustness

5. **Outlier Handling Philosophy** (IQR clip > hard drop):
   *   Tính Q1, Q3, IQR; xác định cận: `lo = Q1 - 1.5×IQR`, `hi = Q3 + 1.5×IQR`
   *   **Clip** giá trị ngoài cận chứ không xóa hàng
   *   Bảo toàn dataset size, kiềm chế nhiễu mà không làm mất thông tin

6. **Data Synchronization in Dyadic Models** (Xóa phải đồng bộ):
   *   Nếu xóa entity, PHẢI xóa tất cả relationships liên quan
   *   Kiểm tra referential integrity: không có relationship mà entity không tồn tại
   *   Tránh data corruption trong các mô hình entity-relationship

### Chi tiết thực hiện:
*   **Làm sạch (Cleaning):** 
    *   Phân biệt Entity vs Relationship; chọn chiến lược xử lý missing phù hợp
    *   Xóa entity nếu quá thiếu (ví dụ: ≥ 5 ô missing) → lưu dropped_ids
    *   Đồng bộ hóa: xóa relationships của entities đã xóa
    *   Điền khuyết: median cho số, mode cho phân loại
    *   Xóa relationships nếu block rating thiếu > 50%
    *   Xóa duplicates
*   **Chuyển đổi (Transformation):** 
    *   **Scaling:** MinMax cho 1-10 scales, Standard cho liên tục (tránh scale trước FE order-dependent)
    *   Mã hóa biến phân loại (One-hot encoding, target encoding)
*   **Xây dựng thuộc tính (Feature Engineering):** 
    *   Trích xuất biến mới (ví dụ: age_gap = |age_1 - age_2| từ raw age TRƯỚC scaling)
    *   Gộp biến có tương quan cao thành feature có ý nghĩa
    *   Tuân thủ DAG của dependencies

## Giai đoạn 4: Lựa chọn và Huấn luyện mô hình (Modeling)

### Chiến lược xử lý mất cân bằng lớp (Class Imbalance):
*   **Nguyên lý từ Reflection:** "Cân bằng khi học, Khách quan khi thi"
    *   KHÔNG thể dùng tỷ lệ 6:2:2 máy móc với dữ liệu lệch
    *   Phải can thiệp vào tập Train để ép mô hình "học" lớp hiếm
    *   Giữ nguyên bản chất "lệch" ở tập Test để đánh giá đúng thực tế
*   **Kỹ thuật áp dụng:**
    *   **Oversampling (SMOTE):** Tăng số mẫu lớp thiểu số trong tập Train
    *   **Undersampling:** Giảm số mẫu lớp đa số trong tập Train
    *   **Class weight:** Gán trọng số cao hơn cho lớp thiểu số
    *   **Threshold adjustment:** Điều chỉnh ngưỡng quyết định thay vì 0.5

### Lựa chọn Thuật toán:
*   **Baseline (Logistic Regression):** 
    *   Tuyến tính, dễ giải thích
    *   Dùng để hiểu baseline performance
*   **Mô hình cơ sở (Decision Tree):** 
    *   Dễ giải thích, không nhạy cảm với scale
    *   Có thể gặp overfitting với dữ liệu phức tạp
*   **Mô hình nâng cao (Gradient Boosting - XGBoost/CatBoost/LightGBM):** 
    *   **Vì sao nên dùng?** (từ Reflection) Trong dữ liệu dạng bảng (Tabular), Gradient Boosting "vua":
        *   Tối ưu hóa hàm loss qua từng bước (iterative optimization)
        *   Xử lý mất cân bằng lớp tuyệt vời (tương tự class_weight)
        *   Feature Importance đáng tin cậy và sắc nét nhất
    *   **Khuyến cáo:** Nhất định so sánh Decision Tree vs Gradient Boosting để tìm winner
*   **Mô hình khác (Random Forest, SVM):** Có thể dùng để so sánh, nhưng thường thua Gradient Boosting trên tabular data

### Hyperparameter Tuning:
*   Sử dụng k-fold Cross-validation (k=3 hoặc 5) trên tập Train
*   GridSearchCV để tìm siêu tham số tối ưu
*   Đánh giá trên Validation set (nếu chia 6:2:2) hoặc CV scores

### Checklist Giai đoạn 4:
- [ ] Kiểm tra Data Leakage: Có dùng đánh giá sau tương tác không?
- [ ] Áp dụng kỹ thuật cân bằng lớp trên tập Train (SMOTE / class_weight)
- [ ] So sánh ít nhất: Baseline → Decision Tree → Gradient Boosting
- [ ] Sử dụng F1-score (không phải Accuracy) làm metric chính
- [ ] Ghi log: mô hình nào, hyperparameter nào, F1 trên Validation là bao nhiêu

## Giai đoạn 5: Đánh giá và Triển khai (Evaluation & Insights)

### Nguyên lý Đánh giá (từ Reflection):

**Accuracy Paradox - "Lời nói dối của độ chính xác":**
*   Trong dữ liệu lệch (match = 1 chỉ 16%), một mô hình đoán bừa lớp 0 vẫn đạt ~84% Accuracy
*   **Kết luận:** Accuracy cao không có nghĩa mô hình tốt với dữ liệu mất cân bằng

**F1-Score - "La bàn thực sự":**
*   F1 = 2 × (Precision × Recall) / (Precision + Recall)
*   Là "la bàn" giúp nhìn thẳng vào hiệu năng lớp thiểu số (match=1)
*   F1 thấp (ví dụ: 0.31) phản ánh độ khó của bài toán: mô hình bỏ lỡ cơ hội (Recall thấp) hoặc dự báo sai (Precision thấp)
*   **Khuyến cáo:** Luôn xem F1-score làm metric chính, không bao giờ tin vào Accuracy một mình

### Đánh giá mô hình:
*   **Chỉ đo lường trên tập Test (chưa bao giờ gặp):**
    *   Accuracy: Tỷ lệ dự báo đúng toàn bộ
    *   **F1-Score:** Cân bằng Precision-Recall (METRIC CHÍNH với dữ liệu lệch)
    *   Precision: Tỷ lệ dự báo "match" là đúng (ít false positive)
    *   Recall: Tỷ lệ bắt được tất cả "match" (ít false negative)
    *   Confusion Matrix: Hiểu rõ TP/FP/TN/FN
*   **Trực quan hóa:**
    *   ROC Curve + AUC: Hiệu năng phân loại trên tất cả threshold
    *   Feature Importance (từ Gradient Boosting): Biến nào quyết định nhất?
    *   Precision-Recall Curve: Tradeoff giữa Precision và Recall

### Rút ra Insights từ Mô hình:
*   **Feature Importance:** Từ Gradient Boosting, xác định top 10 features
    *   Những yếu tố nào thực sự quyết định `match`?
    *   So sánh với giả thuyết ban đầu (Static vs Dynamic, age_gap, ...)?
*   **Giải thích hiện tượng thực tế:**
    *   Confirmed H1: Nghịch lý lựa chọn (condtn=2 match thấp hơn)
    *   Confirmed H2: Yếu tố quyết định nào (attr, shar, age_diff, ...)?
    *   Confirmed H3: age_gap có ảnh hưởng không?
*   **Ghi nhận những "Aha Moments":**
    *   Khoảng cách giữa "mong đợi" (Static) vs "thực tế" (Dynamic) rating
    *   Biến tự tạo (age_gap) có sức mạnh hơn các biến có sẵn

### Checklist Giai đoạn 5:
- [ ] F1-score (không Accuracy) là metric chính để báo cáo
- [ ] Tính Precision, Recall để hiểu error pattern
- [ ] Vẽ ROC curve + lấy AUC
- [ ] Phân tích Feature Importance từ mô hình tốt nhất
- [ ] Kiểm chứng lại các giả thuyết (H1, H2, H3) qua mô hình
- [ ] Ghi lại "Aha Moments" - những phát hiện bất ngờ

---

## Nguyên lý Xuyên Suốt (Cross-cutting Principles)

Những nguyên lý sau áp dụng trên tất cả 5 giai đoạn và bảo vệ toàn bộ dự án khỏi sai lầm phổ biến:

### 1. **Data Leakage - "Thông tin rò rỉ"**
*   **Định nghĩa:** Sử dụng thông tin từ sau tương tác (Future information) để dự báo `match`
*   **Ví dụ:**
    *   SAIT data (ratings sau tương tác) không được dùng trong mô hình
    *   Wave 2 surveys (khảo sát sau Speed Dating) là thông tin lỏn
*   **Kiểm tra:** Nhớ rằng `match` được xác định **tại thời điểm tương tác (wave 1)**, không phải sau
*   **Hậu quả:** Nếu có leakage, mô hình trong thực tế sẽ kém tệ hơn training

### 2. **Entity-Relationship Data Model**
*   **Static (Thực thể):** Thông tin của cá nhân (age, gender, hobbies, income) - cố định, không phụ thuộc quan hệ
*   **Dynamic (Quan hệ):** Thông tin của cặp (match, ratings, age_gap) - nảy sinh từ sự gặp gỡ
*   **Ứng dụng:**
    *   **Cleaning:** Entity → Imputation; Relationship → Deletion
    *   **Scaling:** Không scale entity trước FE order-dependent
    *   **Modeling:** Kết hợp cả static và dynamic features để tối ưu
*   **Lợi ích:** Tránh mất dữ liệu vô tội, giữ tính xác thực của relationship data

### 3. **Domain Knowledge > Máy móc**
*   **Standardization vs Automation:** Học máy nhạy cảm với thang đo, nhưng không phải mọi cột đều cần một cách scaling
*   **Meaningful Aggregation > PCA:** Gộp 17 hobbies thành 5 nhóm có ý nghĩa tốt hơn PCA vô hồn
*   **Order-Dependent FE:** Một biến có thể phụ thuộc vào biến khác → phải tuân thứ tự DAG
*   **Kết luận:** Luôn hỏi "Dữ liệu này có ý nghĩa gì?" trước khi quyết định xử lý

### 4. **Test Set là "Thế giới thực"**
*   **Huấn luyện:** Có thể can thiệp (SMOTE, class_weight, threshold tuning)
*   **Kiểm thử:** Phải giữ nguyên bản chất thực tế (lệch nhãn, outliers, ...)
*   **Nguyên lý:** Bảo vệ tính "khách quan" của Test set → mô hình đánh giá công bằng
*   **Kết quả:** F1-score trên Test sẽ thấp hơn Train (bình thường, đó là dấu tốt)

### 5. **Checklist Nguyên lý Xuyên Suốt**
- [ ] Có dùng bất kỳ future information nào không? (Data Leakage check)
- [ ] Entity vs Relationship data được xử lý khác nhau không?
- [ ] Scaling strategy có dựa trên bản chất dữ liệu không (không phải máy móc)?
- [ ] Feature Engineering có tuân thứ tự DAG không?
- [ ] Test set có được giữ nguyên (không apply SMOTE, threshold tuning)?
- [ ] Sử dụng F1-score (không Accuracy) để so sánh mô hình?
