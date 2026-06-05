# Task 13: Phân tích Hành vi Đa chiều (Comprehensive Behavioral Analytics)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Phân tích các xu hướng tâm lý học dựa trên khoảng cách giữa Kỳ vọng vs. Thực tế và Tự nhận thức vs. Đánh giá khách quan trên toàn bộ **6 thuộc tính cốt lõi**: `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`.
*   **Giai đoạn:** Giai đoạn 5+ (Behavioral Insights for Reporting).
*   **Giả thuyết:** 
    1. Mức độ "thất vọng" (Expectation Gap) sẽ khác nhau tùy thuộc vào thuộc tính (ví dụ: khắt khe hơn với ngoại hình nhưng dễ dãi hơn với trí tuệ).
    2. Mức độ "ngộ nhận" (Misconception Gap) về sự chân thành và thông minh của bản thân thường cao hơn so với ngoại hình.
    3. Tồn tại sự khác biệt giới tính rõ rệt trong cách tự đánh giá và đánh giá đối phương trên 6 phương diện này.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/13_behavioral_analysis.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/13_Behavioral_Analysis.md`).
    *   Biểu đồ Dashboard hành vi: `plots/behavioral_gaps_comprehensive.png`.

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** 
    *   Nhóm Kỳ vọng (Time 1): `attr1_1` đến `shar1_1`.
    *   Nhóm Thực tế (Scorecard): `attr` đến `shar`.
    *   Nhóm Tự nhận thức (Time 1): `attr5_1` đến `amb5_1`.
    *   Nhóm Đối phương chấm (Scorecard): `attr_o` đến `shar_o`.
*   **Phương pháp:** 
    1.  **Chuẩn hóa Thang đo:** Đưa tất cả các biến về thang điểm 10. Đặc biệt xử lý các Wave dùng thang 100 cho nhóm Preference.
    2.  **Tính toán Gap ma trận:** 
        *   `Disappointment_Matrix = Preferences - Reality_Ratings`.
        *   `Misconception_Matrix = Self_Guess - Partner_Ratings`.
    3.  **Trực quan hóa Đa chiều:** Sử dụng **Grouped Bar Chart** hoặc **Heatmap** để so sánh 6 thuộc tính cạnh nhau theo giới tính.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **Bước 1: Chuẩn bị dữ liệu sạch**
    *   Load dữ liệu, trích xuất 24 biến liên quan đến 6 thuộc tính.
- [ ] **Bước 2: Pipeline chuẩn hóa thang đo**
    *   Xử lý đồng nhất thang 10 cho toàn bộ tập dữ liệu.
- [ ] **Bước 3: Phân tích Sự thất vọng (Disappointment Analysis)**
    *   So sánh Kỳ vọng (Tôi muốn gì) vs. Thực tế (Tôi thấy gì ở họ) trên 6 thuộc tính.
- [ ] **Bước 4: Phân tích Sự ngộ nhận (Misconception Analysis)**
    *   So sánh Tự đánh giá (Tôi nghĩ mình thế nào) vs. Khách quan (Họ thấy tôi thế nào) trên 6 thuộc tính.
- [ ] **Bước 5: Dashboard trực quan hóa tổng thể**

## 5. Nhật ký thực thi (Execution Log)
*   *Đang chờ thực hiện...*

## 6. Kết quả & Kiểm chứng (Validation)
*   *Đang chờ kết quả...*

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   *Đang chờ khám phá...*

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Yêu cầu:** Cập nhật các phát hiện về "Bản đồ ngộ nhận con người" vào `Logs/Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   Hoàn thiện chương "Behavioral Economics of Love" trong báo cáo.
