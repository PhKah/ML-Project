# Task 08: Phân tích Phản sự thực & Giải mã Hộp đen AI (Counterfactual & Inverse Scaling)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Sử dụng Phân tích Phản sự thực (Counterfactual) để khám phá các ngưỡng quyết định nhạy cảm của AI.
    2. **[ĐỘT PHÁ]** Áp dụng kỹ thuật **Inverse Scaling (Giải chuẩn hóa)** để dịch ngôn ngữ của AI (Z-score) về các giá trị thực tế của con người (số năm, thang điểm).
    3. Xác nhận tính logic và khả năng giải thích (Explainability) của mô hình XGBoost.
*   **Giai đoạn:** Giai đoạn 6 (Optimization & Refinement) - Vòng lặp tri thức cuối cùng.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `models/winner_model.joblib` (XGBoost), `Data/data_final_v2.csv` (159 cột, đã Deduplicated).
*   **Mã nguồn:** `src/08_counterfactual_analysis.py`.
*   **Đầu ra:** Biểu đồ độ nhạy cảm đã giải mã `plots/counterfactual_scenarios.png`.

## 3. Chiến lược thực hiện (Strategy)
1.  **Mô phỏng động (Dynamic Simulation):** Giữ nguyên một cặp đôi (Pair) gần ngưỡng Threshold nhất, sau đó lần lượt thay đổi các biến số cốt lõi (`age_gap`, `int_corr`) để xem khi nào AI "đổi ý" (chuyển từ No Match sang Match).
2.  **Giải mã thực tế (Inverse Scaling):** Tự động load dữ liệu thô (Raw Data) để tính Mean và Std của các biến tương ứng, từ đó giải chuẩn hóa trục X trên biểu đồ về lại số đo gốc.
3.  **Tối ưu Tri thức:** Tìm ra những **Điểm bùng phát (Tipping Points)** thực sự mang ý nghĩa xã hội học.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] Tạo kịch bản mô phỏng cho Age Gap, Interest Correlation, Surplus.
- [x] Áp dụng Inverse Scaling.
- [x] Xuất biểu đồ và tìm ngưỡng cắt.

## 5. Nhật ký thực thi (Execution Log)
*   **04/06/2026**: Phát hiện lỗi "Mốc 0.00 ảo" do sử dụng dữ liệu Standard Scaled. Quyết định cập nhật mã nguồn để tải thông số Mean/Std từ dữ liệu thô phục vụ việc Inverse Scaling.
*   **04/06/2026**: Thực thi mô phỏng lại với mô hình **XGBoost Winner** trên tập dữ liệu đã loại bỏ đối xứng (Deduplicated). Kết quả trả về vô cùng trực quan.

## 6. Kết quả & Kiểm chứng (Validation)

### 📊 Giải mã Điểm bùng phát (Tipping Points) thực tế:
1.  **Khoảng cách tuổi tác (Age Gap):** Điểm bùng phát nằm ở mốc **~1.92 năm**. Khác với những lầm tưởng ban đầu, XGBoost vô cùng nhạy bén với sự chênh lệch tuổi tác. Khi khoảng cách tuổi vượt qua ranh giới xấp xỉ 2 năm, xác suất Match sụt giảm và không thể vượt qua ngưỡng an toàn. Điều này phản ánh tính chất khắt khe của môi trường hẹn hò đại học, nơi sự đồng trang lứa đóng vai trò cực kỳ quan trọng.
2.  **Tương quan Sở thích (Interest Correlation):** Điểm bùng phát nằm ở mức **0.30**. Mốc này giữ vững tính ổn định ngay cả khi mô hình XGBoost phải làm việc với bộ dữ liệu khó hơn (không có rò rỉ). Nó chứng minh rằng, trong tiềm thức con người, khi sự đồng điệu tổng thể về sở thích đạt ngưỡng 0.30, nó sẽ kích hoạt một lực hút đủ lớn để chuyển từ trạng thái 'do dự' sang trạng thái 'đồng ý'.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   **Sự nhạy bén của XGBoost (Depth-wise Sensitivity):** Mô hình XGBoost (Winner) chứng minh được khả năng đánh giá rủi ro cực tốt. Nó nhận ra rằng một chênh lệch tuổi tác nhỏ (~2 năm) đã đủ để tạo ra sự khác biệt về tâm lý thế hệ trong tập sinh viên. 
*   **Logic Thặng dư (Surplus):** Việc chuyển từ `abs diff` sang Phép trừ có dấu (`Self_Guess - Partner_Expectation`) đã giúp AI xây dựng được các đường ranh giới sắc nét hơn về vị thế "Cửa trên/Cửa dưới" trong một mối quan hệ.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Bài học cốt lõi:** Không bao giờ đọc biểu đồ Counterfactual trực tiếp trên dữ liệu Scaled. Luôn phải có một bước "Việt hóa" (Inverse Scaling) để biến con số thống kê thành các Insight hành vi con người.

## 9. Bước tiếp theo
*   **HOÀN THIỆN BÁO CÁO CUỐI CÙNG (Đã tích hợp).**
