# Task 05: Đánh giá mô hình & Chẩn đoán lỗi (Final Evaluation & 5-Level Diagnostics)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng hệ thống 5 cấp độ kiểm tra sai lầm để đánh giá toàn diện hiệu năng và sự ổn định của hệ thống dự đoán.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation & Insights) theo `plan.md` và `ml_error_detection_standard.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   Hệ thống có bị "ốm" (Overfitting) không?
    *   Các mẫu dữ liệu bị đoán sai (False Negatives) có đặc điểm chung gì?
    *   Mô hình có sẵn sàng cho vận hành thực tế không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** Mô hình mạnh nhất từ Task 04 và tập dữ liệu Test.
*   **Mã nguồn:** `src/05_evaluation.py`
*   **Đầu ra:** 
    *   File log này (`Logs/05_Evaluation_and_Insights.md`).
    *   Biểu đồ **Learning Curves** (Diagnostics Level 2).
    *   **Heatmap Confusion Matrix** (Diagnostics Level 1/3).
    *   Danh sách các mẫu dữ liệu lỗi (Diagnostics Level 3).

## 3. Chiến lược thực hiện (Strategy)
*   **Level 1 (Bề nổi):** Tính F1, Precision, Recall cho tập Test.
*   **Level 2 (Cấu trúc):** Vẽ Learning Curves để kiểm tra Bias-Variance.
*   **Level 3 (Ẩn sâu):** Lọc các trường hợp False Positives và False Negatives để phân tích nguyên nhân tại sao máy đoán sai.
*   **Level 4 (Gốc rễ):** Kiểm tra xem các mẫu sai có phải do outliers hay missing data xử lý chưa tốt không.
*   **Level 5 (Vận hành):** Phân tích hiệu năng theo các phân khúc (Sub-groups) như Gender hoặc Race để đảm bảo tính công bằng (Fairness).

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **Bước 1: Metrics Audit (Level 1)**
    *   Tính toán bộ chỉ số trên tập Test.
- [ ] **Bước 2: Bias-Variance Check (Level 2)**
    *   Sử dụng `learning_curve` từ sklearn.
- [ ] **Bước 3: Error Surgery (Level 3)**
    *   In ra các trường hợp match thực tế nhưng máy bảo không match. Phân tích Profile của họ.

## 5. Nhật ký thực thi (Execution Log)
*(Sẽ được cập nhật sau khi phẫu thuật lỗi)*

## 6. Kết quả & Kiểm chứng (Validation)
*   **Cấp độ 1 & 2:** Tóm tắt tình trạng sức khỏe hệ thống.
*   **Cấp độ 3:** Lý do phổ biến nhất khiến mô hình đoán sai là gì?

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   **Phát hiện:** Những điểm mù (Blind spots) của mô hình.
*   **Kết luận:** Chấp nhận hay bác bỏ các giả thuyết H1, H2, H3.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Yêu cầu:** Cập nhật trọn bộ Insight đắt giá vào `Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   Hoàn thiện Báo cáo PDF cuối cùng.
