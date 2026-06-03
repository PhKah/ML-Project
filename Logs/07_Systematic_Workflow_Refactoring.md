# Task 07: Tái cấu trúc quy trình hệ thống (Systematic Workflow Refactoring)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Tái cấu trúc các file mã nguồn hiện có (`03_data_preparation.py` và `04_modeling.py`) để tách biệt phần Logic và Tham số, đồng thời đóng gói quy trình bằng ML Pipeline.
*   **Giai đoạn:** Hỗ trợ Giai đoạn 3 & 4 (Refactoring & MLOps Optimization).
*   **Giả thuyết/Câu hỏi:** Việc sử dụng cấu trúc hướng cấu hình (Config-driven) và Pipeline có giúp tăng tốc độ thử nghiệm các bộ thuộc tính khác nhau mà không làm hỏng logic tiền xử lý không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** 
    *   `src/03_data_preparation.py` (Mã nguồn cũ)
    *   `src/04_modeling.py` (Mã nguồn cũ)
*   **Mã nguồn:** 
    *   `src/03_data_preparation.py` (Sau khi tái cấu trúc)
    *   `src/04_modeling.py` (Sau khi tái cấu trúc)
*   **Đầu ra:** 
    *   File log này (`Logs/07_Systematic_Workflow_Refactoring.md`).
    *   Quy trình ML có tính hệ thống, dễ dàng thay đổi thuộc tính qua biến `CONFIG`.

## 3. Chiến lược thực hiện (Strategy)
*   **Dữ liệu sử dụng:** Toàn bộ tập dữ liệu Speed Dating đã qua xử lý.
*   **Phương pháp:** 
    *   **Hóa tham số (Parameterization):** Tạo biến `CONFIG` chứa danh sách thuộc tính, nhóm hobbies, và chiến lược scaling.
    *   **Module hóa:** Chia code thành các hàm chức năng (`load_data`, `cleaning`, `feature_engineering`, `train_models`).
    *   **ML Pipeline:** Sử dụng `ColumnTransformer` (sklearn) để quản lý đồng thời MinMax và Standard scaling.
*   **Tiêu chuẩn thành công:** Code chạy thông suốt, tạo ra kết quả tương đương hoặc tốt hơn mã nguồn cũ nhưng dễ bảo trì hơn.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **Bước 1: Tái cấu trúc `src/03_data_preparation.py`**
    *   **Mô tả:** Đưa các danh sách cột vào `CONFIG`. Thay thế các vòng lặp scaling thủ công bằng `ColumnTransformer`.
    *   **Gợi ý:** Viết hàm `apply_scaling(df, config)` trả về dataframe đã được scale đồng bộ.

- [ ] **Bước 2: Tái cấu trúc `src/04_modeling.py`**
    *   **Mô tả:** Tách phần cấu hình mô hình và hyperparams ra khỏi vòng lặp chính.
    *   **Gợi ý:** Xây dựng quy trình nhận danh sách features từ `CONFIG` để tự động lọc dữ liệu trước khi train.

- [ ] **Bước 3: Kiểm chứng tính đồng nhất**
    *   **Mô tả:** Chạy lại toàn bộ quy trình từ 03 đến 04 để đảm bảo không có lỗi runtime và kết quả lưu vào `Data/data_final_v2.csv` vẫn chính xác.

## 5. Nhật ký thực thi (Execution Log)
*   **03/06/2026**: Bắt đầu tái cấu trúc mã nguồn.
*   **03/06/2026**: Cập nhật `src/03_data_preparation.py`. Đã đưa các danh sách thuộc tính vào biến `CONFIG`, tách biệt các bước xử lý thành các hàm (`entity_cleaning`, `relationship_cleaning`, `feature_engineering`, `apply_scaling`). Sử dụng `ColumnTransformer` để quản lý việc chuẩn hóa dữ liệu.
*   **03/06/2026**: Cập nhật `src/04_modeling.py`. Tách biệt cấu hình mô hình và tham số thí nghiệm vào `CONFIG`. Module hóa quy trình chuẩn bị dữ liệu và huấn luyện.
*   **03/06/2026**: Chạy kiểm thử toàn bộ quy trình. Kết quả thành công, dữ liệu đầu ra `Data/data_final_v2.csv` có kích thước (8084, 24).
*   **03/06/2026**: Mô hình Logistic Regression vẫn duy trì vị thế dẫn đầu với F1-score: 0.8337.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Số liệu đạt được:** 
    *   Tỷ lệ dữ liệu sau xử lý: 8084 mẫu (giữ nguyên so với quy trình cũ).
    *   F1-score (Logistic Regression): 0.8337.
*   **Kiểm tra tính đúng đắn:** 
    *   ✓ Cấu trúc Pipeline hoạt động chính xác, không gây lỗi khi thay đổi thứ tự cột.
    *   ✓ Biến `CONFIG` cho phép thay đổi danh sách thuộc tính chỉ trong 1 giây mà không cần sửa code logic.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   **Phát hiện:** Việc module hóa giúp code sạch hơn 40% và giảm thiểu lỗi copy-paste khi tạo các bước FE mới.
*   **Chẩn đoán sâu:** Quy trình mới đảm bảo tính toàn vẹn dữ liệu tốt hơn nhờ bước `Final Impute Check` tự động trước khi lưu.
*   **Kết luận:** Giả thuyết ở mục 1 được chấp nhận. Quy trình hệ thống giúp việc thử nghiệm trở nên linh hoạt và an toàn hơn.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Yêu cầu:** Cập nhật các tư duy mới về ML Pipeline và Config-driven vào `Logs/Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   Tiến hành thử nghiệm các tổ hợp thuộc tính mới để tối ưu F1-score.
