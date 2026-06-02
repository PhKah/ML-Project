# Task 02: EDA chuyên sâu & Kiểm chứng giả thuyết (Advanced EDA & Hypothesis Testing)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Khám phá mối quan hệ nhân quả giữa các biến và biến mục tiêu, đồng thời kiểm chứng các giả thuyết khoa học.
*   **Giai đoạn:** Giai đoạn 2 (Data Understanding) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   **H1 (Choice Overload):** Liệu tỷ lệ `match` ở `condtn=2` (lựa chọn nhiều) có thấp hơn đáng kể so với `condtn=1` không?
    *   **H2 (Static vs Dynamic):** Các biến chấm điểm sau cuộc gặp (`attr`, `fun`,...) có tương quan with `match` mạnh hơn các biến sở thích ban đầu (`attr1_1`,...) không? (Kiểm soát Data Leakage).
    *   **H3 (Age Gap):** Độ lệch tuổi tác giữa hai người có ảnh hưởng đến khả năng `match` không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/02_advanced_eda.py`
*   **Đầu ra:** 
    *   File log này (`Logs/02_Advanced_EDA_and_Hypothesis_Testing.md`).
    *   Các biểu đồ tương quan (Heatmap), biểu đồ phân phối (Boxplot/Bar chart) lưu tại thư mục `plots/`.

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Các biến từ `bare_minimum.md` và biến `age` của đối phương (`age_o`).
*   **Phương pháp:** 
    *   **Tương quan:** Sử dụng Pearson correlation.
    *   **Kiểm định H1:** Tính tỷ lệ `match` trung bình theo `condtn`.
    *   **Phân tích độ tuổi:** Tính toán `age_diff = abs(age - age_o)`.
*   **Tiêu chuẩn thành công:** Xác định được các đặc trưng (features) quan trọng nhất và xác nhận được sự tồn tại của hiện tượng Nghịch lý lựa chọn.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Phân tích Nghịch lý lựa chọn (Choice Overload)**
- [x] **Bước 2: So sánh Tương quan (Static vs Dynamic)**
- [x] **Bước 3: Phân tích chênh lệch tuổi tác**

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành các bước kiểm định giả thuyết.
*   **Chi tiết:** Thực hiện trên môi trường Conda `course`. Các biến đã được tính toán và kiểm chứng thành công.

## 6. Kết quả & Kiểm chứng (Validation)
*   **H1 (Choice Overload):**
    *   `condtn=1` (Ít lựa chọn): Tỷ lệ match ~**20.2%**
    *   `condtn=2` (Nhiều lựa chọn): Tỷ lệ match ~**15.7%**
    *   **Kết luận:** Có sự giảm đáng kể (~4.5%) về tỷ lệ match khi số lượng lựa chọn tăng. Giả thuyết **Nghịch lý lựa chọn** được xác nhận mạnh mẽ.
*   **H2 (Static vs Dynamic Correlation):**
    *   Biến tĩnh (Preferences): Tương quan cực thấp (mạnh nhất là `fun1_1` với 0.04).
    *   Biến động (Ratings): Tương quan cao hơn rõ rệt: `fun` (0.28), `shar` (0.27), `attr` (0.26).
    *   **Kết luận:** Xác nhận hiện tượng **Data Leakage**. Những gì đối phương chấm điểm cho mình sau cuộc gặp quyết định khả năng match hơn nhiều so với sở thích khai báo ban đầu.
*   **H3 (Age Gap):**
    *   Nhóm `match=0`: Chênh lệch tuổi trung bình là **3.75** năm.
    *   Nhóm `match=1`: Chênh lệch tuổi trung bình là **3.18** năm.
    *   **Kết luận:** Những người match với nhau có xu hướng gần tuổi nhau hơn (chênh lệch ít hơn khoảng 0.5 năm).

## 7. Khám phá quan trọng (Insights & Insights Update)
*   **Phát hiện 1:** "Nghịch lý lựa chọn" là có thật trong Speed Dating. Càng nhiều người để chọn, người ta càng ít có xu hướng đồng ý.
*   **Phát hiện 2:** Khoảng cách giữa "Mong đợi" (Static) và "Thực tế" (Dynamic) là rất lớn. Các đặc trưng động (`fun`, `shar`, `attr`) sẽ là "key features" cho mô hình.
*   **Phát hiện 3:** `fun` (Vui vẻ) và `shar` (Sở sở thích chung) có tương quan with `match` cao hơn cả `attr` (Hấp dẫn) trong nhóm biến động.

## 8. Bước tiếp theo
*   Chuyển sang Giai đoạn 3: Tiền xử lý dữ liệu.
    *   Xử lý giá trị thiếu (Imputation).
    *   Xử lý biến phân loại (Encoding).
    *   Chuẩn hóa dữ liệu (Standardization).
