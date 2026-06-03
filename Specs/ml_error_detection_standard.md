# Tiêu chuẩn Kiểm tra và Chẩn đoán hệ thống Học máy (ML Diagnostics & Quality Standard)

Tài liệu này thiết lập quy trình 5 cấp độ để kiểm tra sai lầm và phát hiện sự kém hiệu quả của một hệ thống Học máy, đảm bảo tính đúng đắn từ thử nghiệm đến vận hành thực tế.

## Cấp độ 1: Bề nổi (Kiểm tra bằng Độ đo và Đánh giá tổng quát)
Đây là lớp kiểm tra đầu tiên để biết hệ thống "có đang ốm hay không".

*   **Nguyên tắc "Không bao giờ tin tưởng tập Train":** Nếu mô hình phán đoán rất tốt trên tập huấn luyện nhưng điểm số trên tập kiểm thử (Test/Validation set) lại thấp, hệ thống đang bị lỗi học vẹt (**Overfitting**).
*   **Đừng để Accuracy (Độ chính xác) đánh lừa:** Trong thực tế, dữ liệu thường mất cân bằng (Imbalanced Data). Dùng Accuracy sẽ che giấu sự kém hiệu quả đối với lớp thiểu số quan trọng.
*   **Cách thức thực hiện:**
    *   Bắt buộc phải vẽ **Ma trận nhầm lẫn (Confusion Matrix)** để xem xét chi tiết mô hình đang phân loại nhầm các lớp như thế nào.
    *   Sử dụng các độ đo khắt khe: **Precision**, **Recall** và trung bình điều hòa **F1-Score**.

## Cấp độ 2: Cấu trúc mô hình (Chẩn đoán bằng Learning Curves)
Xác định lỗi do mô hình quá đơn giản (Bias cao) hay quá nhạy cảm (Variance cao).

*   **Cách thức:** Vẽ **Đường cong học tập (Learning Curves)** biểu diễn hiệu năng của tập Train và Validation dựa trên kích thước dữ liệu.
*   **Bắt bệnh Underfitting (Bias cao):** Cả hai đường cong hội tụ ở mức điểm thấp (tỷ lệ lỗi cao). Mô hình không học được quy luật.
    *   *Sửa lỗi:* Chọn mô hình phức tạp hơn, thêm thuộc tính (Feature Engineering), hoặc giảm hiệu chỉnh (Regularization).
*   **Bắt bệnh Overfitting (Variance cao):** Điểm tập Train rất cao nhưng có khoảng cách (gap) lớn so với tập Validation. Mô hình quá nhạy cảm với nhiễu.
    *   *Sửa lỗi:* Đơn giản hóa mô hình, tăng dữ liệu huấn luyện, hoặc tăng cường hiệu chỉnh (L1, L2 penalty).

## Cấp độ 3: Ẩn sâu bên trong (Phân tích Lỗi - Error Analysis)
Nhìn thẳng vào những gì máy tính đang làm sai thay vì chỉ nhìn vào con số thống kê.

*   **Cách thức:**
    *   Trực quan hóa Ma trận nhầm lẫn dưới dạng **Heatmap**, chia tỷ lệ để làm nổi bật các ô chứa lỗi cao nhất.
    *   **Phẫu thuật lỗi:** Lọc riêng các điểm dữ liệu bị đoán sai (False Positives và False Negatives) và in trực tiếp hồ sơ/mẫu dữ liệu đó để kiểm tra bằng tay.
*   **Hành động:** Tìm lý do tại sao mô hình bối rối (ví dụ: thiếu thông tin quan trọng hoặc thuộc tính gây nhiễu) để thực hiện Feature Engineering có định hướng.

## Cấp độ 4: Gốc rễ (Chất lượng dữ liệu đầu vào - GIGO)
Nguyên lý **GIGO (Garbage In, Garbage Out)**: Mọi thuật toán đều thất bại nếu dữ liệu đầu vào không sạch.

*   **Kiểm tra Nhiễu (Noise) & Ngoại lai (Outlier):** Dữ liệu rác hoặc điểm dị biệt có thể phá vỡ các mô hình nhạy cảm như K-Means hoặc Linear Regression.
*   **Kiểm tra Giá trị khuyết thiếu (Missing values):** Đảm bảo chiến lược điền khuyết (Imputation) phù hợp (Mean/Median/Mode) thay vì xóa bỏ máy móc gây mất thông tin.
*   **Sự nhất quán (Normalization):** Đảm bảo đơn vị đo đồng nhất qua quá trình chuẩn hóa, tránh việc các thuộc tính có thang đo lớn lấn át thuộc tính nhỏ.

## Cấp độ 5: Vận hành thực tiễn (Giám sát suy giảm hệ thống)
Phòng ngừa thất bại khi đưa mô hình từ phòng thí nghiệm ra môi trường thực tế.

*   **Lỗi suy giảm dữ liệu (Data Rot / Stale Data):** Xu hướng thế giới thực thay đổi làm dữ liệu cũ lỗi thời, khiến hiệu năng giảm sút theo thời gian.
*   **Lỗi hệ thống đầu vào:** Cảm biến hỏng hoặc lỗi logic hệ thống đẩy dữ liệu bất thường vào mô hình.
*   **Phòng ngừa:**
    *   Thiết lập code **giám sát hiệu năng trực tiếp (Live monitor)** và cảnh báo tự động khi độ chính xác giảm đột ngột.
    *   Xây dựng **luồng phản hồi (Feedback pipeline)** để con người đánh giá lại kết quả.
    *   Định kỳ huấn luyện lại (**Retrain**) mô hình với dữ liệu mới nhất.

---
**Tóm lại:** Nếu hệ thống gặp vấn đề, hãy đi từ Cấp độ 1 & 2 để định vị loại lỗi. Sau đó, "phẫu thuật" từng mẫu sai ở Cấp độ 3 & 4. Cuối cùng, bảo vệ hệ thống bằng Cấp độ 5 để sinh tồn trong thực tế.
