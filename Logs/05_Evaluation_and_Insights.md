# Task 05: Đánh giá mô hình và Trích xuất Tri thức (Evaluation & Insights)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Kiểm định năng lực thực tế của mô hình trên tập Test ẩn hoàn toàn sạch rò rỉ dữ liệu.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation) - Phiên bản "Bọc thép".
*   **Mô hình kiểm định:** **XGBoost** (Winner sau khi Refactor Pipeline).
*   **Ngưỡng quyết định (Threshold):** 0.28 (Tối ưu cho F0.5).

## 2. Kết quả kiểm định trên tập Test (Final Performance)

### 📊 Chỉ số chiến lược
| Chỉ số | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **Precision** | **29.71%** | Cứ 10 gợi ý Match thì có ~3 cặp thành công thật. |
| **Recall** | **30.37%** | Tìm được ~30% tổng số các cặp Match có trong thực tế. |
| **F0.5-Score** | **0.2984** | Ưu tiên tính chính xác của gợi ý lên hàng đầu. |
| **ROC-AUC** | **0.6435** | Khả năng phân loại tốt hơn ngẫu nhiên đáng kể (0.5). |

### 🔍 Phẫu thuật sai số (Error Surgery)
*   **Báo động giả (FP - 194):** Hệ thống vẫn tạo ra tiếng ồn, nhưng ở mức chấp nhận được trong ứng dụng hẹn hò (User thà xem thừa còn hơn bỏ sót hoàn toàn).
*   **Bỏ sót (FN - 188):** Cho thấy tình cảm con người vẫn còn những yếu tố "xác suất" nằm ngoài các biến số khảo sát.

## 3. Khám phá tri thức (Key Insights)

### 🔍 3.1. Sự sụp đổ của "Điểm số ảo"
*   Việc di chuyển Scaling vào Pipeline làm giảm Precision từ 45% xuống 30%. Đây không phải là thất bại, mà là **Sự thật**. Nó chứng minh rằng các mô hình trước đây đã "học thuộc lòng" phân phối dữ liệu thay vì học quy luật tâm lý.

### 🔍 3.2. Tính công bằng giới tính
*   Mô hình hoạt động ổn định trên cả Nam và Nữ (F0.5 lệch nhau < 2%). Điều này đảm bảo trải nghiệm người dùng đồng nhất cho mọi giới tính.

## 4. Đánh giá & Đề xuất giải pháp (Critique & Solutions)

### 🔴 Vấn đề tồn tại
1.  **Độ nhiễu (Noise):** Tỷ lệ FP/TP vẫn là ~2.3. Hệ thống cần khắt khe hơn.
2.  **Tính tuyến tính:** Các đặc trưng Surplus hiện tại chưa bắt được các hiệu ứng "quá nhanh quá nguy hiểm" (khi một chỉ số quá cao lại gây tác dụng ngược).

### 🟢 Giải pháp đề xuất (Roadmap v2.0)
1.  **Threshold Hardening:** Đẩy Threshold lên **0.40** để ép Precision vượt mức 40%. Điều này chấp nhận hy sinh Recall để lấy sự "độc bản" cho gợi ý.
2.  **Feature Interaction:** Bổ sung các biến tương tác bậc cao như `Hobby_Gap * Mutual_Surplus` để tìm ra các nhóm người có chung sở thích và đồng điệu về kỳ vọng.
3.  **Relative Positioning:** Tính toán điểm số của Partner so với trung bình của toàn bộ Wave để bắt trọn hiệu ứng "Nổi bật trong đám đông".

## 5. Kết luận
Mô hình hiện tại đã đạt trạng thái **"Khoa học dữ liệu chuẩn mực"**. Dù điểm số không cao bằng các phiên bản lỗi trước đó, nhưng đây là nền tảng vững chắc và trung thực nhất để triển khai vào thực tế.
