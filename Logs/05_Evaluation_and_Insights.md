# Task 05: Đánh giá hệ thống & Chẩn đoán 5 cấp độ (Final Evaluation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Giải mã tập dữ liệu ẩn (**Hidden Test Set**) để đánh giá khách quan cuối cùng hiệu năng của mô hình.
    2. Áp dụng hệ thống chẩn đoán lỗi 5 cấp độ cho mô hình chiến thắng (**LightGBM**).
    3. Kiểm chứng các giả thuyết nghiên cứu (H1, H2, H3) dựa trên kết quả thực tế.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation & Insights) - Về đích.
*   **Giả thuyết/Câu hỏi:** 
    * Mô hình LightGBM có duy trì được F1-score trên tập Test tương đương với tập Validation (không bị overfitting)?
    * Các đặc trưng Granular (sở thích rời rạc) có thực sự đóng góp vào việc phân loại các mẫu khó không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** 
    *   Mô hình đã khóa: `models/winner_model.joblib` (**LightGBM**).
    *   Tập Test ẩn: `Data/test_set_hidden.csv`.
*   **Mã nguồn:** `src/05_evaluation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/05_Evaluation_and_Insights.md`).
    *   Dashboard chẩn đoán: `plots/evaluation_diagnostics_final.png`.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ khung chẩn đoán 5 cấp độ:
1.  **Bề nổi (Level 1):** Báo cáo Accuracy, Precision, Recall, F1 trên tập Test "tươi".
2.  **Cấu trúc (Level 2):** Kiểm tra sự chênh lệch (Gap) giữa Val và Test để xác nhận tính ổn định.
3.  **Ẩn sâu (Level 3):** Phân tích Ma trận nhầm lẫn (Confusion Matrix) để hiểu sai số.
4.  **Gốc rễ (Level 4):** Soi xét các đặc trưng dẫn đến các cú "Match" bất ngờ hoặc bỏ lỡ.
5.  **Vận hành (Level 5):** Đánh giá ý nghĩa thực tiễn của các Insight (Top Features).

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Tải mô hình và Dữ liệu Test**
- [x] **Bước 2: Chạy dự báo một lần duy nhất**
- [x] **Bước 3: Tổng hợp Metrics**
- [x] **Bước 4: Trích xuất Insight cuối cùng**

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
