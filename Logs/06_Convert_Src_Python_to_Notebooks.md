# Task 06: Convert Src Python Files to Notebooks

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Chuyển đổi các file Python trong thư mục `src/` thành Jupyter Notebook nhằm tạo sự thuận tiện khi đọc, chạy và trình bày.
*   **Giai đoạn:** Hỗ trợ quá trình triển khai và báo cáo, phù hợp với phần công cụ/Documentation.
*   **Giả thuyết/Câu hỏi:** Việc lưu mã thành notebook giúp dễ kiểm tra bước từng bước và thuận tiện cho người hướng dẫn hoặc người đánh giá.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `src/01_initial_eda.py`, `src/02_advanced_eda.py`, `src/03_data_preparation.py`, `src/04_modeling.py`, `src/05_evaluation.py`
*   **Mã nguồn:** N/A (chỉ thực hiện chuyển đổi định dạng)
*   **Đầu ra:** `src/01_initial_eda.ipynb`, `src/02_advanced_eda.ipynb`, `src/03_data_preparation.ipynb`, `src/04_modeling.ipynb`, `src/05_evaluation.ipynb`

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Nội dung của các file Python trong `src/`.
*   **Phương pháp:** Chuyển từng file Python thành notebook chứa một ô code duy nhất với toàn bộ nội dung script.
*   **Tiêu chuẩn thành công:** Mỗi file `.py` được chuyển thành `.ipynb` tương ứng và có thể mở được trong Jupyter/VS Code Notebook editor.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)
- [x] **Bước 1: Kiểm tra thư mục `src/`**
    *   **Mục tiêu cụ thể:** Xác nhận danh sách file Python cần chuyển.
    *   **Hướng dẫn chi tiết:** Dùng `ls src/` để liệt kê file.
- [x] **Bước 2: Chuyển đổi mỗi file Python thành Notebook**
    *   **Mục tiêu cụ thể:** Tạo file `.ipynb` tương ứng cho mỗi file `.py`.
    *   **Hướng dẫn chi tiết:** Sử dụng Python để đọc nội dung `.py`, tạo cấu trúc notebook và ghi ra file `.ipynb`.
- [x] **Bước 3: Kiểm tra kết quả**
    *   **Mục tiêu cụ thể:** Đảm bảo các file `.ipynb` mới được tạo và có thể mở trong môi trường Notebook.
    *   **Hướng dẫn chi tiết:** Mở `src/*.ipynb` bằng VS Code hoặc Jupyter Lab.

## 5. Nhật ký thực thi (Execution Log)
*   **Ngày thực hiện:** 02/06/2026
*   **Trạng thái:** Hoàn thành.
*   **Chi tiết:** Đã chuyển 5 file Python trong `src/` thành notebook. Do môi trường không có `nbformat`, quá trình tạo notebook được thực hiện bằng cách xuất trực tiếp JSON notebook.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Số lượng file tạo:** 5 file notebook.
*   **File output:**
    *   `src/01_initial_eda.ipynb`
    *   `src/02_advanced_eda.ipynb`
    *   `src/03_data_preparation.ipynb`
    *   `src/04_modeling.ipynb`
    *   `src/05_evaluation.ipynb`
*   **Kiểm tra tính đúng đắn:** Các notebook được tạo với định dạng `nbformat` 4 và có cấu trúc ô code hợp lệ.

## 7. Khám phá quan trọng (Insights & Insights Update)
*   Việc chuyển đổi này giúp mã nguồn dễ chia sẻ và chạy từng bước trong môi trường Notebook.
*   Nếu cần, có thể tiếp tục chia nhỏ notebook ra nhiều ô (code, markdown) để trình bày rõ hơn từng phần.

## 8. Bước tiếp theo
*   Mở các file notebook mới trong VS Code và kiểm thử từng ô.
*   Nếu muốn, tách notebook thành các ô code + markdown để tăng tính trình bày và giải thích cho mỗi bước phân tích.
