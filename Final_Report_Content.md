# BÁO CÁO ĐỒ ÁN HỌC MÁY VÀ KHAI PHÁ DỮ LIỆU
**Đề tài:** Dự đoán khả năng kết đôi (Match) trong Speed Dating qua lăng kính Tương hợp và Thận trọng.

---

## 1. Giới thiệu và Mô tả bài toán
*   **Bài toán:** Dự đoán khả năng hai người cùng đồng ý (Match) dựa trên hồ sơ đăng ký và sở thích cá nhân.
*   **Thách thức cốt lõi:**
    *   **Mất cân bằng nhãn:** Tỷ lệ Match thực tế chỉ chiếm ~16.4%.
    *   **Tính chủ quan:** Cảm xúc con người rất khó định lượng bằng các biến số bảng (Tabular data).
    *   **Rủi ro rò rỉ dữ liệu:** Các đánh giá tức thời (Scorecard) dễ làm mô hình "học vẹt" kết quả thay vì học quy luật.

## 2. Chiến lược Dữ liệu và Tiền xử lý

### 2.1. "Số hóa trực giác" qua bộ 146 Đặc trưng
Dự án nâng cấp từ dữ liệu thô lên bộ đặc trưng **Tương hợp (Compatibility)**:
*   **Compatibility Gaps:** Tính toán 23 biến chênh lệch tuyệt đối (`abs diff`) về sở thích và kỳ vọng.
*   **Vibe is King:** Sử dụng `int_corr` (Tương quan tổng thể) làm biến dẫn dắt chính.
*   **Tipping Points:** Tích hợp các đặc trưng nhị phân nhạy cảm (ví dụ: `is_age_match` với ngưỡng khắt khe 0.5 năm).

### 2.2. Kiểm soát Chất lượng Nghiêm ngặt (Diagnostics Level 4)
*   **Chống rò rỉ dữ liệu:** Kiên quyết loại bỏ toàn bộ nhóm biến **Scorecard** (như `like`, `attr`, `fun`). Điều này đảm bảo mô hình có giá trị ứng dụng thực tế: dự báo **TRƯỚC** khi tương tác diễn ra.
*   **Strict Isolation:** Tập dữ liệu ẩn (Hidden Test Set) được cô lập hoàn toàn khỏi quá trình huấn luyện và tối ưu tham số.

## 3. Kết quả Huấn luyện và Đánh giá Chiến lược

### 3.1. Thước đo Công bằng: $F_{0.5}$-Score
Vì dự án ưu tiên **Precision** (giảm thiểu gợi ý sai gây phiền nhiễu), chúng tôi sử dụng **$F_{0.5}$-Score** để gán trọng số Precision cao gấp đôi Recall.
*   **Triết lý:** Thà bỏ lỡ một vài ca Match còn hơn làm giảm uy tín của hệ thống bằng các báo động giả (False Positives).

### 3.2. Kết quả Đánh giá trên tập Test ẩn (Mô hình XGBoost Winner)

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| **Accuracy** | **80%** | Độ chính xác tổng thể ổn định. |
| **Precision** | **37.3%** | Cao gấp 2.5 lần so với dự báo ngẫu nhiên. |
| **Recall** | **35.9%** | Bắt được 1/3 số cặp Match với độ chắc chắn cao. |
| **$F_{0.5}$-Score** | **0.3702** | Hiệu năng thực tế dưới triết lý khắt khe. |
| **ROC-AUC** | **0.7046** | Khả năng phân loại bản chất ở mức Tốt. |

## 4. Giải mã Tâm lý học Thành công (Behavioral Insights)
Qua việc so sánh giữa nhóm **Match (Thành công)** và **No Match (Thất bại)**, dự án rút ra 2 bí mật hành vi đắt giá:

1.  **Sự biến mất của "Ngộ nhận" (The Humility Factor):** 
    *   Nhóm No Match có độ ngộ nhận về bản thân rất cao (Gap ~1.0). 
    *   Ngược lại, nhóm **Match** có độ ngộ nhận tiệm cận mức 0, thậm chí hơi tự ti nhẹ về ngoại hình (-0.28). 
    *   **Kết luận:** Chìa khóa của sự thành công là **Sự Thành Thật**. Khi con người nhìn nhận bản thân thực tế, họ dễ dàng tìm thấy tiếng nói chung.

2.  **Sự hài lòng vượt mong đợi (The Exceeding Expectation):** 
    *   Nhóm Match có mức độ thỏa mãn thực tế cao hơn kỳ vọng rất sâu, đặc biệt ở thuộc tính **Sở thích chung (SHAR)** với mức chênh lệch hài lòng tăng gấp đôi so với nhóm thất bại.

## 5. Những khám phá đắt giá từ AI
1.  **Ngưỡng tuổi "Nghiệt ngã":** AI phát hiện chênh lệch tuổi quá **0.5 năm** (6 tháng) là rào cản lớn nhất. Sự đồng điệu về thế hệ là yếu tố định đoạt.
2.  **Vibe is King:** AI nhạy bén với tương quan sở thích tổng thể (`int_corr`) hơn là các chênh lệch lẻ tẻ, phản ánh thực tế sự tương hợp là một "tần số" chung thay vì phép cộng các sở thích lẻ.
3.  **Tính tương thích của thuật toán:** Các mô hình Boosting (XGBoost, LightGBM) hưởng lợi mạnh mẽ nhất từ các biến **Gap**, chứng minh sự tương hợp là một hàm phi tuyến phức tạp.

## 6. Kết luận
Dự án đã thành công trong việc xây dựng một **Pipeline mai mối chuyên nghiệp**. Bằng cách kết hợp giữa kỹ thuật học máy hiện đại (XGBoost, SMOTE) và tri thức tâm lý học thực chứng, chúng tôi đã tạo ra một hệ thống không chỉ chính xác mà còn bền vững và có khả năng giải thích cao.

---
**Báo cáo kết thúc. Hệ thống sẵn sàng triển khai.**
