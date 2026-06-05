# Task 05: Đánh giá hệ thống & Chẩn đoán 5 cấp độ (Final Evaluation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Giải mã tập dữ liệu ẩn (**Hidden Test Set**) để đánh giá khách quan cuối cùng hiệu năng của mô hình.
    2. Áp dụng hệ thống chẩn đoán lỗi 5 cấp độ cho mô hình chiến thắng (**XGBoost**).
    3. **[KIẾN TRÚC MỚI]** Cấy ghép các tiêu chuẩn đánh giá nâng cao (PR-AUC, Learning Curve) từ phiên bản V1 để tăng cường độ tin cậy khoa học.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation & Insights) - Về đích.
*   **Giả thuyết/Câu hỏi:** 
    * Mô hình có duy trì được F1-score trên tập Test tương đương với tập Validation (không bị overfitting)?
    * Thước đo PR-AUC (Average Precision) có chứng minh được sức mạnh thực sự của mô hình trên lớp thiểu số (Match=1) không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** 
    *   Mô hình đã khóa: `models/winner_model.joblib`.
    *   Tập Test ẩn: `Data/test_set_hidden.csv`.
*   **Mã nguồn:** `src/05_evaluation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/05_Evaluation_and_Insights.md`).
    *   Dashboard chẩn đoán: `plots/evaluation_diagnostics_final.png`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ khung chẩn đoán 5 cấp độ:
1.  **Bề nổi (Level 1):** Báo cáo Accuracy, Precision, Recall, F1 trên tập Test "tươi".
2.  **Cấu trúc (Level 2):** **[CẢI TIẾN]** Sử dụng **PR-AUC (Average Precision)** làm thước đo công bằng cho dữ liệu mất cân bằng.
3.  **Ẩn sâu (Level 3):** Phân tích Ma trận nhầm lẫn (Confusion Matrix) để hiểu sai số.
4.  **Gốc rễ (Level 4):** Soi xét các đặc trưng dẫn đến các cú "Match" bất ngờ hoặc bỏ lỡ.
5.  **Vận hành (Level 5):** **[CẢI TIẾN]** Tích hợp biểu đồ **Learning Curve** (Đường cong học tập) để chẩn đoán mức độ Overfit/Underfit một cách trực quan, làm bằng chứng thép bảo vệ mô hình.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **[KIẾN TRÚC MỚI] Bước 1: Tích hợp Metric PR-AUC (Average Precision)** vào báo cáo Level 1.
- [ ] **[KIẾN TRÚC MỚI] Bước 2: Bổ sung biểu đồ Learning Curve** vào Dashboard Diagnostic Level 5.
- [x] **Bước 3: Tải mô hình và Dữ liệu Test**
- [x] **Bước 4: Chạy dự báo một lần duy nhất**
- [x] **Bước 5: Tổng hợp Metrics và Insight**

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 6: Final Decoding & Diagnostics
*   *Mục tiêu: Đã giải mã tập Test ẩn (N=1642). Mô hình LightGBM cho kết quả thực tế cực kỳ ổn định.*

#### **CẤP ĐỘ 1: Metrics trên tập TEST (Kết quả cuối cùng)**

| Metric | Giá trị | Giải thích |
|--------|---------|------------|
| **F1-score (Match)** | **0.3658** | Hiệu năng thực chất sau khi loại bỏ mọi rò rỉ dữ liệu. |
| **Precision** | 0.28 | 1 trong 4 dự báo "Match" là chính xác trong thực tế. |
| **Recall** | **0.52** | Bắt được hơn một nửa (52%) các cặp đôi thực sự khớp. |
| **ROC-AUC** | 0.6726 | Khả năng phân loại khá tốt trên dữ liệu chưa bao giờ gặp. |
| **Accuracy** | 0.70 | Độ chính xác tổng thể (không nên dùng làm metric chính). |

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Chẩn đoán Overfitting & Tính ổn định
*   **Gap Analysis:** Điểm Val F1 (0.413) và Test F1 (0.366) có sự chênh lệch nhỏ (0.047). Điều này cho thấy mô hình **không bị Overfitting** nghiêm trọng và có khả năng tổng quát hóa tốt.
*   **Tính nhất quán:** Mô hình duy trì được Recall > 0.5 trên cả hai tập, chứng tỏ "luật hấp dẫn" mà nó học được là bền vững.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### 🩺 Chẩn đoán cấp sâu (Level 3-5)

*   **Level 3 (Error Surgery):** 
    *   **FP (360):** "Báo động giả" là vấn đề chính. Mô hình có xu hướng lạc quan.
    *   **FN (129):** Bỏ lỡ 129 cơ hội Match. Đây là chi phí cơ hội cho người dùng.
*   **Level 4 (Outlier rate):** Tỷ lệ ngoại lệ ở các mẫu dự báo sai (0.09) thấp hơn mẫu đúng (0.11), chứng tỏ lỗi không phải do dữ liệu rác mà do bản chất bài toán "Pre-match" rất khó đoán định chỉ qua hồ sơ tĩnh.
*   **Level 5 (Fairness):** 
    *   F1-score Nữ: 0.3651
    *   F1-score Nam: 0.3664
    *   **Kết luận:** Mô hình cực kỳ **CÔNG BẰNG** về mặt giới tính (chênh lệch chỉ 0.0013).

### 💡 Insight Cuối Cùng
1.  **Age Gap (Top 1):** Khoảng cách tuổi vẫn là bộ lọc mạnh nhất.
2.  **Sapiosexual Signal:** Trí tuệ tự thân (`intel3_1`) vượt qua ngoại hình để trở thành yếu tố dự báo quan trọng.
3.  **Lifestyle Match:** Việc `tvsports_o` lọt vào top feature cho thấy thói quen giải trí thụ động của đối phương có ảnh hưởng lớn đến quyết định cá nhân.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Cập nhật:** Kết quả này là con số chính thức để đưa vào báo cáo cuối kỳ. Pipeline SMOTE + LightGBM Dyadic là cấu hình tối ưu nhất cho dự án này.

## 9. Bước tiếp theo
*   **Hoàn thiện Final Report Content.**
