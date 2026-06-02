# Task 03: Chuẩn bị và Tiền xử lý dữ liệu (Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Làm sạch và chuyển đổi dữ liệu thô thành tập dữ liệu sẵn sàng cho huấn luyện mô hình.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    *   Làm thế nào để xử lý giá trị khuyết thiếu mà không làm mất quá nhiều thông tin?
    *   Việc chuẩn hóa dữ liệu có giúp cân bằng tầm quan trọng của các biến có thang đo khác nhau không?
    *   Việc trích xuất đặc trưng mới (`age_diff`) có mang lại giá trị dự báo cao hơn không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu đã làm sạch (`Data/data_cleaned.csv`).
    *   Tập dữ liệu đã chuẩn hóa (`Data/data_final.csv`).

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Các biến trong `bare_minimum.md` và `age_o`.
*   **Phương pháp:** 
    *   **Lọc cột:** Chỉ giữ lại các biến "Sống còn" và biến mục tiêu.
    *   **Xử lý Missing Values:** 
        *   Sử dụng **Median** cho các biến số (Age, Ratings) để tránh ảnh hưởng của ngoại lai.
        *   Sử dụng **Mode** (Giá trị phổ biến nhất) cho các biến phân loại (Race).
    *   **Feature Engineering:** Tạo biến `age_diff = abs(age - age_o)`.
    *   **Mã hóa (Encoding):** Sử dụng One-hot Encoding cho biến `race`.
    *   **Chuẩn hóa (Transformation):** Sử dụng **StandardScaler** (Z-score) cho các biến số.
*   **Tiêu chuẩn thành công:** Không còn giá trị thiếu trong tập dữ liệu cuối cùng; dữ liệu được đưa về cùng thang đo (trung bình 0, phương sai 1).

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Trích lọc và Xử lý Missing Values**
- [x] **Bước 2: Feature Engineering & Encoding**
- [x] **Bước 3: Chuẩn hóa dữ liệu (Standardization)**

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành.
*   **Mã nguồn:** `src/03_data_preparation.py`
*   **Chi tiết:** 
    *   Đã cài đặt `scikit-learn` cho môi trường Conda `course`.
    *   Dữ liệu được làm sạch, trích xuất `age_diff`, mã hóa `race` (One-hot) và chuẩn hóa bằng `StandardScaler`.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Số lượng bản ghi:** 8,378 (giữ nguyên so với ban đầu).
*   **Số lượng thuộc tính:** 23 (bao gồm các cột dummy của `race`).
*   **Kiểm tra giá trị thiếu:** 0 (Đã điền Median/Mode cho toàn bộ các ô trống).
*   **Đầu ra:** 
    *   `Data/data_cleaned.csv`: Dữ liệu đã làm sạch, chưa chuẩn hóa.
    *   `Data/data_final.csv`: Dữ liệu cuối cùng (đã chuẩn hóa), sẵn sàng cho Modeling.

## 7. Khám phá quan trọng (Insights & Insights Update)
*   **Phát hiện 1:** Việc sử dụng Median giúp bảo toàn kích thước tập dữ liệu mà không làm lệch phân phối của các biến quan trọng như `age`.
*   **Phát hiện 2:** Sau khi chuẩn hóa, tất cả các thuộc tính số đều có cùng thang đo, điều này đặc biệt quan trọng cho các thuật toán như SVM hay Logistic Regression.

## 8. Bước tiếp theo
*   Chuyển sang Giai đoạn 4: Lựa chọn và Huấn luyện mô hình (Modeling).
