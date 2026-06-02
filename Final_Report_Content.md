# NỘI DUNG BÁO CÁO ĐỒ ÁN HỌC MÁY VÀ KHAI PHÁ DỮ LIỆU

## 1. Giới thiệu và Mô tả bài toán
*   **Bài toán:** Dự đoán khả năng kết đôi (Match) trong các sự kiện Speed Dating dựa trên dữ liệu nhân khẩu học, sở thích cá nhân và đánh giá sau cuộc gặp.
*   **Tầm quan trọng:** Hiểu được các quy luật tâm lý và hành vi ẩn sau quyết định chọn bạn đời của con người, từ đó ứng dụng vào việc tối ưu hóa các thuật toán gợi ý của ứng dụng hẹn hò.
*   **Loại bài toán:** Phân loại nhị phân (Binary Classification).

## 2. Phương pháp học máy và Dữ liệu
*   **Tập dữ liệu:** Speed Dating Data (Columbia University) với 8,378 bản ghi.
*   **Tiền xử lý (Data Preparation):**
    *   Xử lý giá trị khuyết thiếu bằng Median (biến số) và Mode (biến phân loại) để bảo toàn kích thước mẫu.
    *   Mã hóa One-hot cho các biến định danh (Race).
    *   Chuẩn hóa Z-score để đưa các thuộc tính về cùng thang đo, tránh việc các thuộc tính lớn lấn át thuộc tính nhỏ.
*   **Thuật toán sử dụng:** So sánh giữa Logistic Regression (Baseline), Decision Tree, Random Forest và SVM.
*   **Chiến lược đánh giá:** Hold-out (60% Train, 20% Val, 20% Test) kết hợp k-fold Cross-validation để tránh Overfitting.

## 3. Kết quả thí nghiệm và Đánh giá
*   **Metric chủ đạo:** F1-Score (do dữ liệu mất cân bằng lớp: chỉ 16.5% match).
*   **Kết quả:** Decision Tree đạt F1-Score cao nhất (0.33) trên tập Validation và được lựa chọn làm mô hình cuối cùng.
*   **Hiệu năng tập Test:** Accuracy 77%, F1-Score 0.31. Tuy con số không quá cao nhưng phản ánh đúng độ phức tạp và tính ngẫu nhiên trong hành vi chọn người yêu của con người.

## 4. Cấu trúc chương trình (Source Code Structure)
*   **Tổ chức theo Task-Oriented:**
    *   `src/01_initial_eda.py`: Khám phá cấu trúc và schema dữ liệu.
    *   `src/02_advanced_eda.py`: Kiểm chứng các giả thuyết khoa học.
    *   `src/03_data_preparation.py`: Quy trình làm sạch và chuẩn hóa dữ liệu.
    *   `src/04_modeling.py`: Huấn luyện, tối ưu siêu tham số (GridSearchCV).
    *   `src/05_evaluation.py`: Đánh giá cuối cùng và trích xuất Insight.

## 5. Khó khăn và Cách giải quyết
*   **Vấn đề Data Leakage:** Ban đầu, các biến chấm điểm sau cuộc gặp có tương quan quá mạnh khiến mô hình bị "lệch".
    *   *Giải quyết:* Thực hiện EDA chuyên sâu để phân tách biến "Tĩnh" và "Động", từ đó có cái nhìn khách quan về sức mạnh dự báo của từng nhóm.
*   **Mất cân bằng lớp (Imbalance):** Lớp match=1 quá ít.
    *   *Giải quyết:* Chuyển đổi tư duy đánh giá từ Accuracy sang F1-score và AUC để mô hình không bị "ngủ quên" trên lớp đa số.

## 6. Quan điểm về Học máy và Kết luận
*   **Insight đắt giá 1 (Nghịch lý lựa chọn):** Học máy đã chứng minh khi con người có quá nhiều lựa chọn, xác suất họ thực sự hài lòng và kết đôi sẽ giảm xuống (~4.5%).
*   **Insight đắt giá 2 (Ngoại hình vs Tính cách):** Dù `attr` (Hấp dẫn) là yếu tố dẫn dắt đầu tiên, nhưng `age_diff` (chênh lệch tuổi) mới là "biến ẩn" có giá trị dự báo cực cao trong Decision Tree.
*   **Triết lý ứng dụng:** Học máy không chỉ là dự đoán đúng 0 hay 1, mà là quá trình thấu hiểu cấu trúc của dữ liệu. Trong bài toán này, khả năng giải thích (Interpretability) của Decision Tree quý giá hơn nhiều so với độ chính xác cao nhưng "đen tối" của các mô hình phức tạp.

**Kết luận:** Dự án đã ứng dụng thành công quy trình học máy bài bản để bóc tách một vấn đề thực tế đầy cảm tính thành những con số và quy luật có thể định lượng.
