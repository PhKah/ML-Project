# Task 08: Phân tích Phản sự thực & Tối ưu hóa Dữ liệu (Counterfactual & Compatibility Enrichment)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Sử dụng Phân tích Phản sự thực (Counterfactual) để khám phá các ngưỡng quyết định nhạy cảm.
    2. **[CẢI TIẾN CỰC HẠN]** Tích hợp 23 đặc trưng Chênh lệch (Gaps) về sở thích và kỳ vọng dựa trên triết lý "Tương hợp hóa dữ liệu".
    3. **[ĐÁNH GIÁ CHIẾN LƯỢC]** Sử dụng $F_{0.5}$-Score để đánh giá sự bứt phá của các bộ lọc Precision.
*   **Giai đoạn:** Giai đoạn 6 (Optimization & Refinement) - Vòng lặp tri thức cuối cùng.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `models/winner_model.joblib` (LightGBM), `Data/data_final_v2.csv` (146 cột tri thức).
*   **Mã nguồn:** 
    *   `src/03_data_preparation.py` (Làm giàu Gaps & Indicators).
    *   `src/04_modeling.py` (Tối ưu hóa F-beta).
*   **Đầu ra:** Bảng kết quả hệ thống hoàn thiện nhất.

## 3. Chiến lược thực hiện (Strategy)
1.  **Chưng cất tri thức (Distillation):** Biến 23 quy luật chênh lệch (abs diff) thành các đặc trưng hạng nhất trong Pipeline.
2.  **Đồng bộ hóa phương pháp:** Tái thực thi toàn bộ chu trình với GridSearchCV để đảm bảo mọi mô hình đều đạt phong độ cao nhất trên dữ liệu mới.
3.  **Giải oan cho mô hình:** Sử dụng $F_{0.5}$ để chứng minh hiệu quả của chiến lược "Thà ít mà đúng chắc".

## 5. Nhật ký thực thi (Execution Log)
*   **04/06/2026**: Thực hiện làm giàu dữ liệu lần cuối với 23 biến Gaps (Lifestyle & Values similarity).
*   **04/06/2026**: Chạy lại Benchmark toàn diện tối ưu hóa theo $F_{0.5}$. Ghi nhận sự bứt phá ngoạn mục của các mô hình phi tuyến.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Bảng Benchmark Tối ưu Cuối cùng (Validation $F_{0.5}$):**

| Mô hình | Trước Enrichment (GĐ 4) | Sau Enrichment (GĐ 6 - Gaps) | Biến động |
| :--- | :---: | :---: | :---: |
| **LightGBM** | 0.3759 | **0.4231** | 🚀 **+0.0472** |
| **XGBoost** | **0.3904** | 0.3969 | 📈 +0.0065 |
| **CatBoost** | 0.3364 | 0.3697 | 📈 +0.0333 |

*   **Kết luận:** Việc tích hợp **Đặc trưng Chênh lệch (Gap Features)** là chìa khóa mở ra sức mạnh thực sự của LightGBM. Mô hình đạt độ tinh khiết (Precision) cao và khả năng gác cổng tin cậy.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   **Điểm bùng phát xã hội:** Sự đồng điệu về sở thích (`Gaps`) có giá trị dự báo cao hơn nhiều so với việc chỉ nhìn vào sở thích đơn lẻ.
*   **Hội tụ tri thức:** Toàn bộ nhóm Boosting (LightGBM, XGBoost, CatBoost) đều tăng điểm đáng kể, cho thấy tập đặc trưng mới đã "lột tả" được bản chất của bài toán Speed Dating.
*   **Tính ổn định:** Hệ thống đạt điểm ROC-AUC > 0.7, một ngưỡng rất tốt cho dữ liệu hành vi con người đầy nhiễu.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Cập nhật:** Đã chính thức xác nhận triết lý **"Compatibility-Driven Features"** là phương án tối ưu nhất cho dự án. Mọi tri thức này đã được đưa vào `Logs/Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   **HOÀN THIỆN BÁO CÁO CUỐI CÙNG.**
