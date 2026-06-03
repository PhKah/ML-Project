# Tiêu chuẩn Thấu hiểu Dữ liệu (Data Understanding Standard)

Dự án tuân thủ 5 yếu tố cốt lõi sau đây trong quá trình phân tích và khai phá dữ liệu:

## 1. Cấu trúc không gian của dữ liệu (Hàng và Cột)
*   **Quan sát (Observations):** Mỗi hàng là một thực thể (véctơ n chiều).
*   **Thuộc tính (Features):** Mỗi cột là một đặc trưng.
*   **X & y:** Xác định rõ ma trận đặc trưng (Input) và mảng mục tiêu (Output).

## 2. Kiểu dữ liệu và Ngữ nghĩa (Data Types & Semantics)
*   **Phân loại:** Định danh (Categorical), Nhị phân (Binary), Số thực (Continuous).
*   **Đơn vị & Thang đo:** Đảm bảo nhất quán về đơn vị đo lường và định dạng (Ngày tháng, tiền tệ...).

## 3. Mức độ khuyết thiếu dữ liệu (Missing Values)
*   **Nhận diện:** NaN, null, NA.
*   **Cơ chế:** Đánh giá lý do khuyết thiếu (lỗi kỹ thuật, chủ ý người dùng, hay bản chất dữ liệu) để chọn phương án xử lý (Imputation/Removal).

## 4. Sự tồn tại của Nhiễu (Noise) và Ngoại lai (Outliers)
*   **Nhiễu:** Sai lệch do nhập liệu hoặc nguồn thu thập không ổn định.
*   **Ngoại lai:** Điểm dữ liệu dị biệt. Xác định đó là lỗi hay là thông tin quan trọng (gian lận, đột biến).

## 5. Sự đa dạng, Phân bố và Tính dư thừa
*   **Phân bố:** Kiểm tra sự đa dạng của thuộc tính (loại bỏ các cột đơn giá trị).
*   **Tính dư thừa:** Sử dụng hệ số tương quan/hiệp phương sai để loại bỏ các cột trùng lặp ý nghĩa, giảm chi phí tính toán.

## 6. Tiêu chuẩn Tiền xử lý dữ liệu (Data Preprocessing Standard)

### 6.1 Chiến lược Scaling
*   **MinMax [0,1]:** Cho dữ liệu có thang đo giới hạn (1-10 scales như hobbies, ratings). Bảo toàn ý nghĩa tỷ lệ.
*   **StandardScaler (Z-score):** Cho dữ liệu liên tục (age, income). Chuẩn hóa về mean=0, std=1 để phù hợp với các mô hình nhạy cảm.
*   **Nguyên tắc:** Không có một cách scaling phổ quát. Scaling strategy phải dựa trên bản chất thang đo của dữ liệu.

### 6.2 Dimensionality Reduction via Meaningful Aggregation
*   **Phương pháp:** Thay vì loại bỏ hoặc PCA máy móc, hãy **gộp có chủ ý** dựa trên correlation hoặc domain knowledge.
*   **Quy trình:**
    1. Tính correlation matrix giữa các biến liên quan
    2. Nhóm các biến có tương quan cao
    3. Tính mean/weighted average cho mỗi nhóm
    4. Thay thế 17 biến → 5 feature "có ý nghĩa"
*   **Lợi ích:** Giảm chiều + tăng khả năng giải thích + tăng robustness.

### 6.3 Order-Dependent Feature Engineering
*   **Nguyên tắc:** Một số biến phụ thuộc vào trạng thái của biến khác. Phải xác định rõ thứ tự.
*   **Ví dụ:** Nếu cần `age_gap = |age(iid) - age(pid)|`, PHẢI tính trước khi scale age.
*   **Cách kiểm tra:** Vẽ DAG (Directed Acyclic Graph) của feature dependencies.

### 6.4 Entity vs Relationship Missing Data Strategy
*   **Entity data (Users, Products...):** Ưu tiên IMPUTATION (điền median/mode)
    *   Lý do: Xóa entity = mất thông tin nội tại về object
*   **Relationship data (Interactions, Transactions...):** Ưu tiên DELETION nếu quá thiếu
    *   Lý do: Mối quan hệ phải "chân thực"; điền trung bình = tạo dữ liệu giả
    *   Ngưỡng suy nghị: Nếu một block (ví dụ: 7 cột rating) thiếu > 50%, xóa hàng
*   **Nguyên tắc:** Hiểu ngữ nghĩa của dữ liệu → chọn chiến lược xử lý phù hợp.

### 6.5 Outlier Handling Philosophy
*   **Phương pháp:** IQR clip (giới hạn) thay vì hard drop
    *   Tính Q1, Q3, IQR = Q3 - Q1
    *   Cận: `lo = Q1 - 1.5×IQR`, `hi = Q3 + 1.5×IQR`
    *   Clip giá trị: `col = col.clip(lo, hi)`
*   **Lợi ích:** Bảo toàn kích thước dataset, không mất hàng, nhưng kiềm chế nhiễu.
*   **Nguyên tắc:** Outlier là "nhiễu" cần kiềm chế, không phải "lỗi" cần xóa.

### 6.6 Data Synchronization in Dyadic/Relational Models
*   **Nguyên tắc:** Xóa một entity PHẢI đồng bộ hóa các relationship liên quan.
*   **Ví dụ:** Nếu xóa 9 user, PHẢI xóa tất cả interaction của 9 người đó.
*   **Kiểm tra:** Referential integrity - không có interaction mà user không tồn tại.
*   **Cách thực hiện:** Lưu dropped_ids từ bước xóa entity, sau đó filter relationship.
