# Task 01: Định hình bài toán & Khám phá dữ liệu sơ bộ (Framing & Initial EDA)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Xác định cụ thể các biến đầu vào/đầu ra và thực hiện kiểm tra sơ bộ cấu trúc dữ liệu theo tiêu chuẩn môn học.
*   **Giai đoạn:** Giai đoạn 1 (Framing) và Giai đoạn 2 (Data Understanding) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   Cấu trúc dữ liệu có khớp với mô tả trong `bare_minimum.md` không?
    *   Tỷ lệ mất cân bằng giữa các lớp (match = 0 vs match = 1) là bao nhiêu? (Điều này sẽ quyết định việc sử dụng F1-score thay vì Accuracy).
    *   Tình trạng khuyết thiếu dữ liệu ở các biến "sống còn" như thế nào để chọn phương pháp xử lý (xóa hàng hay điền Mean/Median)?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`
*   **Mã nguồn:** `src/01_initial_eda.py`
*   **Đầu ra:** 
    *   File log này (`Logs/01_Framing_and_Initial_EDA.md`).
    *   Các thông số thống kê mô tả (info, missing values, class distribution).

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Toàn bộ tập dữ liệu, tập trung vào các biến trong `bare_minimum.md`.
*   **Phương pháp:** 
    *   Sử dụng `pandas` để load và kiểm tra schema.
    *   Thống kê số lượng và tỷ lệ % giá trị thiếu cho từng biến.
    *   Trực quan hóa sự phân bổ của biến mục tiêu `match`.
*   **Tiêu chuẩn thành công:** Xác định được danh sách các cột cần giữ lại và chiến lược làm sạch dữ liệu cụ thể cho GĐ3.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **Bước 1: Load dữ liệu và Kiểm tra Schema cơ bản**
    *   **Mục tiêu cụ thể:** Đảm bảo file được đọc đúng định dạng và nhận diện được số lượng hàng/cột.
    *   **Hướng dẫn chi tiết:** Sử dụng `pd.read_csv` với encoding `ISO-8859-1`.
    *   **Gợi ý Code:**
        ```python
        import pandas as pd
        df = pd.read_csv('Data/Speed Dating Data.csv', encoding='ISO-8859-1')
        print(f"Kích thước: {df.shape}")
        df.info()
        ```

- [ ] **Bước 2: Phân tích Biến mục tiêu (Target Analysis)**
    *   **Mục tiêu cụ thể:** Kiểm tra sự cân bằng của lớp `match`.
    *   **Hướng dẫn chi tiết:** Đếm số lượng giá trị 0 và 1.
    *   **Gợi ý Code:**
        ```python
        print(df['match'].value_counts(normalize=True))
        ```

- [ ] **Bước 3: Kiểm tra dữ liệu thiếu cho các biến "Sống còn"**
    *   **Mục tiêu cụ thể:** Xác định mức độ thiếu hụt để lên phương án xử lý (Bỏ bản ghi hay điền Mean/Median).
    *   **Gợi ý Code:**
        ```python
        essential_vars = ['gender', 'age', 'race', 'imprace', 'imprelig', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'condtn', 'match']
        missing_stats = df[essential_vars].isnull().mean() * 100
        print(missing_stats.sort_values(ascending=False))
        ```

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành các bước phân tích sơ bộ.
*   **Chi tiết:**
    *   Đã cài đặt `pandas` và `numpy` cho môi trường thực thi.
    *   Dữ liệu được load thành công với encoding `ISO-8859-1`.
    *   Kích thước tập dữ liệu: 8,378 dòng và 195 cột.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Phân bổ biến mục tiêu (match):**
    *   `match = 0`: 6,998 (83.53%)
    *   `match = 1`: 1,380 (16.47%)
    *   **Kết luận:** Dữ liệu mất cân bằng lớp rõ rệt. Cần sử dụng F1-score và các kỹ thuật xử lý mất cân bằng ở GĐ4.
*   **Tình trạng dữ liệu thiếu (Missing values):**
    *   Các biến chấm điểm sau cuộc gặp (`shar`, `amb`) có tỷ lệ thiếu cao nhất (~8.5% - 12.7%).
    *   Các biến cá nhân (`gender`, `condtn`, `match`) hoàn toàn đầy đủ.
    *   Các biến khác (`age`, `race`, `attr1_1`, ...) thiếu dưới 1.5%.
*   **Độ tuổi:** Phổ rộng từ 18 đến 55 tuổi, trung bình là 26.3.

## 7. Khám phá quan trọng (Insights & Insights Update)
*   **Phát hiện 1:** Tỷ lệ thành công (match) khá thấp (16.5%), phản ánh thực tế của Speed Dating.
*   **Phát hiện 2:** Có sự "Rò rỉ dữ liệu" (Data Leakage) tiềm ẩn nếu dùng các biến chấm điểm sau cuộc gặp để dự báo. Cần cân nhắc tách biệt hai nhóm biến này trong báo cáo.
*   **Phát hiện 3:** Các biến về sở thích (`attr1_1`, ...) có sự đa dạng cao, cần chuẩn hóa (Standardization) ở GĐ3.

## 8. Bước tiếp theo
*   Chuyển sang Giai đoạn 2: EDA chuyên sâu.
    *   Trực quan hóa mối tương quan giữa các biến chấm điểm và `match`.
    *   Phân tích ảnh hưởng của `condtn` (số lượng lựa chọn) đến tỷ lệ `match` (Nghịch lý lựa chọn).
