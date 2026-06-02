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
