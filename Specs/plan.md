# Project Implementation Framework (Framework-First Approach)

Kế hoạch này tuân thủ nguyên tắc **"Dữ liệu quyết định Công cụ"**, bao gồm 5 giai đoạn cốt lõi của vòng đời Khoa học Dữ liệu, bám sát giáo trình môn học.

## Giai đoạn 1: Định hình bài toán (Framing & Approach)
*   **Hiểu rõ nhu cầu thực tế (Business understanding):** Dự đoán khả năng "Match" giữa hai người trong sự kiện Speed Dating.
*   **Đưa về bài toán Học máy (Analytic approach):**
    *   **Loại bài toán:** Học có giám sát (Supervised Learning) - **Phân loại nhị phân (Binary Classification)**.
    *   **Không gian bài toán:** Xác định tập đặc trưng đầu vào ($X$) và nhãn đầu ra ($y$ - `match`).

## Giai đoạn 2: Xử lý Dữ liệu thô (Data Acquisition & Understanding)
*   **Thu thập (Acquisition):** Sử dụng tập dữ liệu Speed Dating (lưu tại `Data/`).
*   **Khám phá và Thấu hiểu (Data Understanding - EDA):** 
    *   Sử dụng thống kê mô tả và trực quan hóa (biểu đồ Histogram, Boxplot, Scatter plot).
    *   Phát hiện các điểm bất thường (anomalies), giá trị khuyết thiếu (missing values), và sự mất cân bằng lớp (class imbalance).

## Giai đoạn 3: Chuẩn bị và Tiền xử lý dữ liệu (Data Preparation)
*   **Làm sạch (Cleaning):** 
    *   Bỏ qua bản ghi (xóa hàng/cột) nếu tỷ lệ khuyết thiếu quá cao.
    *   Sử dụng hằng số toàn cục (Global constant) hoặc điền bằng đại lượng trung bình (Mean)/trung vị (Median).
    *   Sử dụng giá trị có xác suất cao nhất (thông qua hồi quy hoặc cây quyết định).
*   **Chuyển đổi (Transformation):** 
    *   Chuẩn hóa dữ liệu (Normalization) bằng Min-Max Scaling hoặc Z-score Standardization.
    *   Rời rạc hóa (Discretization) hoặc mã hóa biến phân loại (Encoding).
*   **Xây dựng thuộc tính (Feature Engineering):** Trích xuất các thuộc tính mới có ý nghĩa thống kê.

## Giai đoạn 4: Lựa chọn và Huấn luyện mô hình (Modeling)
*   **Chia dữ liệu (Hold-out):** Tách tập dữ liệu thành 3 phần: Huấn luyện (60%), Tối ưu/Validation (20%), và Kiểm thử (20%).
*   **Thuật toán:**
    *   Mô hình cơ sở (Baseline): Logistic Regression.
    *   Mô hình phức tạp: Cây quyết định (Decision Tree), Rừng ngẫu nhiên (Random Forest), và Máy vectơ hỗ trợ (SVM).
*   **Lựa chọn tham số (Hyperparameter Tuning):** Sử dụng tập Validation hoặc k-fold Cross-validation trên tập huấn luyện để tìm bộ siêu tham số tối ưu (ví dụ: tham số C, kernel trong SVM, độ sâu của cây).

## Giai đoạn 5: Đánh giá và Triển khai (Evaluation & Insights)
*   **Đánh giá mô hình:** Chỉ đo lường hiệu năng trên tập Kiểm thử (Test set).
    *   Các độ đo: **Accuracy, Confusion Matrix, Precision, Recall, F1-score**.
    *   Trực quan hóa: Biểu đồ đường cong ROC và chỉ số AUC.
*   **Trình bày (Insights):** Rút ra tri thức về Feature Importance và giải thích các hiện tượng thực tế (ví dụ: Nghịch lý lựa chọn).
