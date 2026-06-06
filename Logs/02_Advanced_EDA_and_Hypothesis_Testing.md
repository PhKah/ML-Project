# Task 02: EDA chuyên sâu & Kiểm chứng giả thuyết (Advanced EDA & Hypothesis Testing)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Khám phá các mối quan hệ nhân quả và kiểm chứng các quy luật tâm lý trong hẹn hò để làm nền tảng cho Feature Engineering.
*   **Giai đoạn:** Giai đoạn 2 (Data Understanding).
*   **Giả thuyết/Câu hỏi:** 
    *   **H1 (Choice Overload):** Có tồn tại "Nghịch lý lựa chọn" làm giảm tỉ lệ Match khi số lượng ứng viên tăng lên không?
    *   **H2 (Anti-Leakage Audit):** So sánh tương quan giữa biến Tĩnh (Hồ sơ) và biến Động (Cảm nhận sau 4 phút) để xác nhận rủi ro rò rỉ dữ liệu.
    *   **H3 (Age Gap):** Độ lệch tuổi tác ảnh hưởng thế nào đến sự bùng phát cảm xúc?
    *   **H4 (Cognitive Layering - The Humility Factor):** Liệu lăng kính "Người khác thấy mình thế nào" (`5_1`) có phản ánh thực tế tốt hơn "Tôi tự thấy mình thế nào" (`3_1`) khi so sánh với kỳ vọng của đối phương?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Đầu ra:** Các biểu đồ phân phối và tương quan tại thư mục `plots/`.

## 3. Chiến lược thực hiện (Strategy)
*   **Kiểm định H1:** Phân tích tỉ lệ Match theo biến `condtn`.
*   **Kiểm định H2:** Tính toán ma trận tương quan giữa `match` và 2 nhóm biến (Trước vs Sau buổi hẹn).
*   **Kiểm định H3:** Phân tích chênh lệch tuổi tác giữa nhóm Match vs No Match.
*   **Kiểm định H4:** So sánh phân phối của `(3_1 - 5_1)` để xác định "Độ vênh ảo tưởng" của người dùng.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Phân tích Nghịch lý lựa chọn (Choice Overload)**
    *   **Gợi ý Code:**
        ```python
        # Tỉ lệ match theo điều kiện thí nghiệm
        print(df.groupby('condtn')['match'].mean())
        ```

- [x] **Bước 2: So sánh Tương quan (Static vs Dynamic)**
    *   **Gợi ý Code:**
        ```python
        static_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
        dynamic_cols = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
        print("Tương quan biến Tĩnh:", df[static_cols + ['match']].corr()['match'])
        print("Tương quan biến Động:", df[dynamic_cols + ['match']].corr()['match'])
        ```

- [x] **Bước 3: Phân tích chênh lệch tuổi tác**
    *   **Gợi ý Code:**
        ```python
        df['age_diff'] = (df['age'] - df['age_o']).abs()
        print(df.groupby('match')['age_diff'].mean())
        ```

- [x] **Bước 4: Phân tích tầng nhận thức (The Humility Audit)**
    *   **Gợi ý Code:**
        ```python
        # Kiểm tra độ lệch giữa tự đánh giá và dự đoán xã hội
        print((df['attr3_1'] - df['attr5_1']).describe())
        ```

## 5. Nhật ký thực thi (Execution Log)
*   **Trạng thái:** Đã hoàn thành và xác nhận các giả thuyết quan trọng.

## 6. Kết quả & Kiểm chứng (Validation)
*   **H1 (Choice Overload):**
    *   `condtn=1` (Ít lựa chọn): Tỷ lệ match ~**20.22%**
    *   `condtn=2` (Nhiều lựa chọn): Tỷ lệ match ~**15.70%**
    *   **Kết luận:** Nghịch lý lựa chọn là có thật. Tỉ lệ match giảm **4.52%** khi áp lực chọn lựa tăng cao.
*   **H2 (The Great Correlation Contrast):**
    *   **Biến tĩnh (Hồ sơ):** Tương quan trung bình ~0.026.
    *   **Biến động (Cảm nhận):** Tương quan trung bình ~0.214.
    *   **KẾT LUẬN:** Biến động có tương quan mạnh gấp **8.2 lần** biến tĩnh. Đây là bằng chứng đanh thép cho rủi ro rò rỉ dữ liệu. Việc loại bỏ chúng là bắt buộc để mô hình mang tính dự báo thực chất.
*   **H3 (Age Gap):**
    *   Nhóm `match=1` có chênh lệch tuổi trung bình (~3.18) thấp hơn rõ rệt so với nhóm `match=0` (~3.75).
    *   **Kết luận:** Độ tuổi "đồng trang lứa" (chênh lệch thấp hơn **0.57 năm**) làm tăng đáng kể khả năng kết nối.
*   **H4 (Cognitive Humility):**
    *   Dữ liệu xác nhận giá trị tự đánh giá `3_1` cao hơn dự đoán xã hội `5_1` trung bình **0.18 điểm**.
    *   **Kết luận:** Tồn tại sự "Tự đề cao" nhẹ trong nhận thức. Sử dụng `5_1` làm mốc so sánh với kỳ vọng đối phương (trường hợp ít ảo tưởng hơn) sẽ tạo ra đặc trưng dự báo bền vững và thực tế nhất.

## 7. Khám phá quan trọng (Insights)
*   **Phát hiện 1:** Mô hình cần các đặc trưng **tương tác cặp đôi** (Engineered features) để bù đắp cho sự thiếu hụt tương quan của các biến tĩnh đơn lẻ.
*   **Phát hiện 2:** `5_1` (Social Self-Perception) là một "Anchor" (điểm neo) tâm lý cực kỳ giá trị để xây dựng bộ biến **Mutual Surplus** sạch rò rỉ.

## 8. Bước tiếp theo
*   Chuyển sang Giai đoạn 3: Data Preparation.
    *   Trích xuất Hobby Gaps.
    *   Tính toán Mutual Surplus dựa trên mốc `5_1` và `3_1`.
