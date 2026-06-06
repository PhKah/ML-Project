# Task 01: Định hình bài toán & Khám phá dữ liệu sơ bộ (Framing & Initial EDA)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Xác định cụ thể các biến đầu vào/đầu ra và thiết lập triết lý dự báo "Bọc thép" (không sử dụng thông tin hậu sự kiện).
*   **Giai đoạn:** Giai đoạn 1 (Framing) và Giai đoạn 2 (Data Understanding).
*   **Triết lý Dự báo:** Chuyển từ "Dự báo dựa trên cảm xúc nhất thời" sang "Dự báo dựa trên bản chất hồ sơ và kỳ vọng". Điều này đảm bảo mô hình có tính ứng dụng thực tế cao (có thể dùng để gợi ý ghép đôi ngay khi người dùng đăng ký).

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`
*   **Đầu ra:** Bản đồ đặc trưng sạch rò rỉ, bảng phân bổ nhãn.

## 3. Chiến lược thực hiện (Strategy)
*   **Phát biểu bài toán:** Phân loại nhị phân dựa trên dữ liệu Dyadic (cặp đôi).
*   **Biến đầu vào (X):** Chỉ giữ lại các biến tĩnh (Nhân khẩu học, Sở thích, Kỳ vọng 1_1, Tự nhận thức 3_1, 5_1).
*   **Biến đầu ra (y):** `match` (0: Không thành, 1: Thành công).
*   **Anti-Leakage Strategy:** Kiên quyết loại bỏ các cột chấm điểm đối phương ngay sau cuộc gặp (`attr`, `sinc`...) vì đây là các biến "biết rồi mới nói", gây rò rỉ thông tin trầm trọng.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Load dữ liệu và Kiểm tra Schema cơ bản**
- [x] **Bước 2: Phân tích Biến mục tiêu (Target Analysis)**
- [x] **Bước 3: Kiểm tra dữ liệu thiếu cho các biến "Sống còn"**

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026 (Cập nhật 06/06/2026)
*   **Kích thước tập dữ liệu:** 8,378 dòng và 195 cột.
*   **Tình trạng:** Phát hiện 7 "bóng ma" thiếu thông tin hồ sơ diện rộng, kéo theo **79 bản ghi tương tác** không hợp lệ.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Phân bổ biến mục tiêu (match):**
    *   `match = 0`: 83.53%
    *   `match = 1`: 16.47%
    *   **Kết luận:** Mất cân bằng lớp nặng. Ưu tiên Precision (F0.5-score) để lọc gợi ý chất lượng.
*   **Tình trạng dữ liệu thiếu (Static Audit):** 
    *   `expnum` thiếu nhiều nhất (**78.5%**).
    *   Nhóm biến tự nhận thức `_5_1` thiếu đồng loạt **41.4%**.
    *   Các biến nhân khẩu học cơ bản (`age`, `gender`) đầy đủ 100%.

## 7. Khám phá quan trọng (Insights)
*   **Phát hiện 1:** Tỉ lệ Match tự nhiên thấp (16.5%) cho thấy bài toán tìm kiếm sự tương hợp là cực kỳ khó khăn.
*   **Phát hiện 2 (CHÍ MẠNG):** Các biến chấm điểm sau cuộc gặp có tương quan ảo với `match`. Nếu đưa vào mô hình, điểm số sẽ rất cao nhưng hệ thống sẽ vô dụng trong thực tế.
*   **Phát hiện 3:** Việc khuyết thiếu dữ liệu ở nhóm `5_1` đòi hỏi chiến lược Imputation cẩn trọng ở GĐ 4 để không làm nhiễu tín hiệu của "Yếu tố Khiêm tốn".

## 8. Bước tiếp theo
*   Tiến hành Advanced EDA để kiểm chứng các giả thuyết về "Nghịch lý lựa chọn" và "Tương phản tương quan".
