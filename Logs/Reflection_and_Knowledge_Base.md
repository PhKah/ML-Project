# TỔNG KẾT TƯ DUY VÀ TRI THỨC HỌC MÁY (PERSONAL REFLECTION)

File này lưu trữ những nguyên lý cốt lõi và tư duy giải quyết vấn đề mà tôi (người thực hiện) đã rút ra được qua dự án này. Đây là "tri thức nền" có thể áp dụng cho mọi bài toán học máy khác.

## 1. Tư duy về Dữ liệu (Data Mindset)
*   **Nguyên lý "Static vs Dynamic":** Trong thực tế, có những thứ ta biết trước (Sở thích) và những thứ chỉ nảy sinh trong quá trình tương tác (Đánh giá). Một mô hình học máy tốt phải phân biệt được hai loại này để tránh **Data Leakage**.
*   **Giá trị của sự khuyết thiếu:** Việc xử lý Missing Values bằng Median/Mode không chỉ là điền vào chỗ trống, mà là sự lựa chọn: Ta chấp nhận một sai số nhỏ để giữ lại bức tranh tổng thể lớn.

## 2. Tư duy về Đánh giá (Evaluation Mindset)
*   **Cái bẫy của Accuracy:** Trong một thế giới mất cân bằng (như việc kết đôi thành công vốn là hiếm hoi), con số 80-90% Accuracy có thể là lời nói dối vĩ đại nhất. **F1-Score** chính là "la bàn" giúp ta nhìn thẳng vào sự thật về hiệu năng của lớp thiểu số (match=1).
*   **Sự cân bằng giữa Complexity & Interpretability:** Một mô hình cực kỳ chính xác nhưng không giải thích được "tại sao" thì cũng giống như một thầy bói. Trong khoa học xã hội, **Decision Tree** quý giá vì nó cho ta thấy rõ những "nhánh rẽ" của quyết định con người.

## 3. Tư duy về Quy trình (Workflow Mindset)
*   **Tiêu chuẩn hóa (Standardization):** Học máy rất nhạy cảm với thang đo. Luôn đặt câu hỏi: "Các thuộc tính của mình đã đứng trên cùng một mặt mặt phẳng chưa?" trước khi đưa vào mô hình.
*   **Chia để trị (Task-Oriented):** Việc đánh số code (`01_`, `02_`,...) và ghi log không phải là thủ tục hành chính, mà là cách để bảo vệ tư duy của mình khỏi sự lộn xộn khi bài toán trở nên phức tạp.

## 4. Những "Aha Moments" từ dự án
*   **Nghịch lý lựa chọn:** Khi có quá nhiều sự lựa chọn, bộ não con người có xu hướng trở nên khắt khe hơn. Điều này đã được chứng minh thực nghiệm qua sự sụt giảm tỷ lệ match trong dữ liệu.
*   **Biến ẩn `age_diff`:** Đôi khi những biến ta tự tạo ra (Feature Engineering) lại có sức mạnh hơn cả những biến có sẵn. Sự tương đồng về tuổi tác là một lực hút thầm lặng nhưng cực kỳ mạnh mẽ.

---
**Tri thức này là của tôi. Tôi đã trải nghiệm nó qua từng dòng code và biểu đồ.**
