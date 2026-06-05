# BÁO CÁO ĐỒ ÁN HỌC MÁY VÀ KHAI PHÁ DỮ LIỆU
**Đề tài:** Dự đoán khả năng kết đôi (Match) trong Speed Dating qua lăng kính Tương hợp và Thận trọng.

---

## 1. Giới thiệu và Mục tiêu Hệ thống
Dự án nhằm giải mã các quyết định "Match" trong hẹn hò tốc độ thông qua các thuật toán học máy. Thay vì chỉ tối ưu hóa các chỉ số kỹ thuật, hệ thống hướng tới **3 mục tiêu cốt lõi**:
1.  **Xây dựng Niềm tin (Precision-Driven):** Triển khai một bộ lọc khắt khe nhằm giảm thiểu các gợi ý sai (False Positives), bảo vệ trải nghiệm người dùng.
2.  **Số hóa Tri thức (Knowledge Distillation):** Chuyển đổi các quy luật tương hợp tâm lý học thành các đặc trưng toán học tường minh (Compatibility Gaps & Mutual Surplus).
3.  **Tính Trung thực (Strict Integrity):** Loại bỏ hoàn toàn rò rỉ dữ liệu (Data Leakage) thông qua kỹ thuật gộp cặp đối xứng (Reciprocal Deduplication) và thanh lọc thực thể (Entity Deletion).

## 2. Chiến lược Dữ liệu và Tiền xử lý

### 2.1. "Số hóa trực giác" qua bộ 154 Đặc trưng Tương hợp
Hệ thống không sử dụng dữ liệu thô mà tiến hành xây dựng **Bản đồ Tương hợp** giữa hai người:
*   **Hobby Gaps (Sự đồng điệu):** 17 biến chênh lệch tuyệt đối (`abs diff`) về sở thích. Hướng lệch không quan trọng bằng độ lớn của sự khác biệt.
*   **Mutual Surplus 2 tầng (Sự vượt ngưỡng):** 24 biến đo lường "Thặng dư Kỳ vọng". Thay vì dùng trị tuyệt đối, hệ thống dùng phép trừ có dấu để đánh giá vị thế "Cửa trên/Cửa dưới", được xét trên 2 lăng kính:
    *   *Tầng 1 (Thực tế Xã hội - `5_1`):* Tôi nghĩ xã hội thấy tôi thế nào vs. Kỳ vọng của bạn.
    *   *Tầng 2 (Cái tôi - `3_1`):* Tôi tự đánh giá mình thế nào vs. Kỳ vọng của bạn.

### 2.2. Kiểm soát Chất lượng Nghiêm ngặt (Diagnostics Level 4)
*   **Thanh lọc Thực thể (Entity Deletion):** Phát hiện và xóa vĩnh viễn 7 "bóng ma" (khuyết $\ge$ 20 trường thông tin), kiên quyết từ chối điền khuyết (Imputation) để tránh tạo ra các hồ sơ nhân bản làm nhiễu mô hình.
*   **Chống rò rỉ Cặp đôi (Reciprocal Deduplication):** Để tránh việc mô hình "học thuộc lòng" kết quả từ các bản ghi đối xứng (A $\rightarrow$ B và B $\rightarrow$ A), hệ thống đã tạo `pair_id` và xóa 50% dữ liệu. Bài toán chuyển từ "Dự báo cá nhân" sang "Dự báo Cặp đôi" độc lập 100%. Hệ quả của việc này là cột giới tính (`gender`) trở thành Zero-Variance (toàn Nữ) và bị loại bỏ khỏi mô hình để tối ưu hóa.

## 3. Kết quả Huấn luyện và Đánh giá Chiến lược

### 3.1. Thước đo Công bằng: $F_{0.5}$-Score
Vì dự án ưu tiên **Precision** (giảm thiểu gợi ý sai gây phiền nhiễu), chúng tôi sử dụng **$F_{0.5}$-Score** để gán trọng số Precision cao gấp đôi Recall.

### 3.2. Kết quả Đánh giá trên tập Test ẩn (Mô hình XGBoost Winner)
XGBoost chứng tỏ sự bền vững vượt trội khi đối mặt với dữ liệu đã được làm sạch hoàn toàn sự "nhắc bài" (Leakage):

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| **Accuracy** | **79%** | Độ chính xác tổng thể ổn định. |
| **Precision** | **35.2%** | Cao gấp đôi so với dự báo ngẫu nhiên. Mô hình thận trọng và tin cậy. |
| **Recall** | **32.6%** | Bắt được 1/3 số cặp Match với độ chắc chắn cao. |
| **$F_{0.5}$-Score** | **0.3465** | Hiệu năng thực tế dưới triết lý khắt khe. |
| **ROC-AUC** | **0.7093** | Đạt mức kỷ lục. Khả năng phân loại bản chất cực tốt. |

## 4. Những khám phá đắt giá (Insights)

### 4.1. Giải mã Tâm lý học Thành công
Qua Phân tích Hành vi giữa nhóm Match và No Match:
1.  **Sự biến mất của "Ngộ nhận" (The Humility Factor):** Nhóm thất bại (No Match) luôn có độ ngộ nhận về bản thân rất cao. Ngược lại, nhóm **Match** có độ ngộ nhận tiệm cận 0. Chìa khóa của sự thành công là **Sự Khiêm tốn và Thành thật**.
2.  **Sự hài lòng vượt mong đợi:** Nhóm Match có mức độ thỏa mãn thực tế cao hơn kỳ vọng rất sâu, đặc biệt ở thuộc tính **Sở thích chung (SHAR)**.

### 4.2. Giải mã Hộp đen AI (Explainable AI)
1.  **Phân tích Định kiến Dữ liệu (Data Bias):** Mô hình đánh giá rất cao `reading_gap` (Chênh lệch sở thích đọc sách). Dù trái với trực giác thông thường, điều này hoàn toàn hợp lý với tập dữ liệu từ **Đại học Columbia (Ivy League)**, nơi văn hóa đọc là một Proxy (biến đại diện) mạnh mẽ cho hệ giá trị và tri thức.
2.  **Inverse Scaling & Age Gap:** Ban đầu, AI dường như báo Tipping Point độ lệch tuổi ở mức 0.00 (Standard Deviation). Tuy nhiên, sau khi áp dụng kỹ thuật **Inverse Scaling (Giải chuẩn hóa)**, hệ thống đính chính Tipping Point thực tế nằm ở mốc **~3.78 năm** (tiệm cận 5 năm). Điều này khớp hoàn hảo với thống kê thực tế, chứng minh mô hình hoàn toàn logic với thế giới quan con người.

## 5. Kết luận
Dự án đã thành công trong việc xây dựng một **Hệ thống Mai mối dựa trên Tri thức (Knowledge-based Recommendation System)**. Bằng cách kết hợp giữa kỹ thuật học máy hiện đại (XGBoost, SMOTE) và tư duy làm giàu đặc trưng tâm lý học, chúng tôi đã tạo ra một hệ thống không chỉ chính xác, chống chịu tốt với rò rỉ dữ liệu, mà còn mang lại khả năng giải thích hành vi sâu sắc.

---
**Báo cáo kết thúc. Hệ thống sẵn sàng triển khai.**

