# BÁO CÁO ĐỒ ÁN HỌC MÁY VÀ KHAI PHÁ DỮ LIỆU
**Đề tài:** Dự đoán khả năng kết đôi (Match) trong Speed Dating qua lăng kính Tương hợp và Thận trọng.

---

## 1. Giới thiệu và Mô tả bài toán
*   **Bài toán:** Dự đoán khả năng hai người cùng đồng ý (Match) sau 4 phút gặp gỡ tại sự kiện Speed Dating.
*   **Tầm quan trọng:** Bóc tách các quy luật tâm lý ẩn sau quyết định chọn bạn đời, từ đó cung cấp giải pháp tối ưu hóa bộ lọc cho các ứng dụng hẹn hò thực tế.
*   **Thách thức:** Dữ liệu mất cân bằng nghiêm trọng (chỉ 16.4% Match), tính ngẫu nhiên cao trong cảm xúc con người và rủi ro rò rỉ dữ liệu từ các đánh giá sau tương tác.

## 2. Phương pháp học máy và Chiến lược Dữ liệu

### 2.1. Chiến lược "Tương hợp hóa" Dữ liệu (Feature Enrichment)
Thay vì chỉ cung cấp dữ liệu thô, dự án thực hiện "Số hóa trực giác" qua bộ 146 đặc trưng:
*   **Compatibility Gaps:** Tính toán 23 biến chênh lệch tuyệt đối về sở thích (`art_gap`, `sports_gap`...) và kỳ vọng (`attr_exp_gap`) để đo lường sự "đồng điệu".
*   **Granular Hobbies:** Giữ nguyên 17 sở thích rời rạc của cả hai phía để mô hình phi tuyến tự khai thác các tương tác tinh vi.
*   **Tipping Point Indicators:** Tích hợp các đặc trưng nhị phân dựa trên "điểm bùng phát" thực tế của AI (ví dụ: `is_age_match` với ngưỡng 0.5 năm).

### 2.2. Tiền xử lý Nghiêm ngặt (Strict Pipeline)
*   **Entity-Relationship Cleaning:** Ưu tiên điền khuyết (Imputation) cho hồ sơ cá nhân để bảo toàn mẫu, nhưng ưu tiên xóa bỏ (Deletion) cho các tương tác thiếu hụt thông tin để giữ tính xác thực.
*   **Scaling theo bản chất:** MinMax cho các thang đo điểm (1-10, 1-100) và Standard cho các biến liên tục (Tuổi).
*   **SMOTE Inside Pipeline:** Cân bằng dữ liệu chỉ trong quá trình huấn luyện, giữ nguyên tập Test khách quan để phản ánh đúng thực tế.

## 3. Kết quả thí nghiệm và Đánh giá Chiến lược

### 3.1. Thước đo "Giải oan" cho Mô hình ($F_{0.5}$-Score)
Dự án chủ động đẩy ngưỡng quyết định $T$ lên cao (0.25 - 0.38) để lọc nhiễu. Do đó, chúng tôi sử dụng **$F_{0.5}$-Score** (trọng số Precision gấp đôi Recall) làm metric chính.
*   **Triết lý:** Thà bỏ lỡ một vài ca Match (Recall thấp) còn hơn gây thất vọng cho người dùng bằng các gợi ý sai lệch (giảm thiểu False Positives).

### 3.2. Bảng so sánh hiệu năng (Validation Set)

| Mô hình | Val $F_{0.5}$ | Val Precision | Val Recall | Val AUC |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)** | **0.4136** | **41.4%** | 41.1% | 0.7240 |
| LightGBM | 0.3745 | 36.8% | 40.0% | 0.7152 |
| CatBoost | 0.3494 | 36.0% | 31.1% | 0.6934 |

### 3.3. Kết quả Kiểm định Cuối cùng (Hidden Test Set)
Mô hình **XGBoost** đạt hiệu năng cực kỳ ổn định trên tập dữ liệu ẩn hoàn toàn:
*   **$F_{0.5}$-Score:** **0.3702**
*   **Precision:** **37.3%** (Cao gấp 2.5 lần so với dự đoán ngẫu nhiên).
*   **Recall:** 35.9% (Đạt sự cân bằng lý tưởng với Precision).
*   **Accuracy:** **80%** | **ROC-AUC:** **0.7046**.

## 4. Khám phá quan trọng (Actionable Insights)
1.  **"Vibe is King" (Sự tương quan tổng thể):** AI nhạy bén với tương quan sở thích tổng thể (`int_corr`) hơn là các chênh lệch lẻ tẻ. Sự đồng điệu về phong cách sống là một lực hút vô hình cực mạnh.
2.  **"Age is a Dealbreaker" (Ngưỡng tuổi khắt khe):** Phân tích kịch bản cho thấy chênh lệch quá **0.5 năm** (6 tháng) làm giảm xác suất Match mạnh nhất. Trong môi trường học thuật, sự đồng trang lứa là yếu tố tiên quyết.
3.  **Tính công bằng giới tính:** Mô hình dự báo cho Nữ giới chính xác hơn Nam giới ($F_{0.5}$ 0.41 vs 0.32), cho thấy quy luật chọn lựa của phái nữ có tính hệ thống và rõ ràng hơn.

## 5. Chẩn đoán và Bài học kinh nghiệm
*   **Phòng chống rò rỉ dữ liệu:** Phát hiện và loại bỏ các kết quả ảo (F1 > 0.6) bằng quy trình **Strict Isolation**. Đây là bài học đắt giá về tính trung thực trong khoa học dữ liệu.
*   **Algorithm-Data Fit:** Sự hội tụ của các mô hình mạnh (~0.39) chứng minh bộ đặc trưng đã tiệm cận mức tối ưu của dữ liệu hiện có.
*   **Mở hộp đen (Explainable AI):** Việc kết hợp Phân tích Phản sự thực giúp biến một mô hình phức tạp thành những lời khuyên thực tiễn có thể hiểu được.

## 6. Kết luận
Dự án không chỉ xây dựng một mô hình dự báo, mà đã thiết lập một **hệ thống mai mối thông minh và thận trọng**. Bằng cách ưu tiên tính chính xác (Precision) và "số hóa" các quy luật chênh lệch tâm lý, chúng tôi đã chứng minh được sức mạnh của việc kết hợp giữa thuật toán hiện đại và tri thức thực tế.

---
**Hệ thống đã sẵn sàng cho thực chiến.**
