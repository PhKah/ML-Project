# Task 08: Phân tích Phản sự thực & Tối ưu hóa Dữ liệu (Counterfactual & Feature Enrichment)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Sử dụng Phân tích Phản sự thực (Counterfactual) để khám phá các ngưỡng quyết định ngầm định của mô hình tốt nhất hiện tại.
    2. **[CẢI TIẾN]** Số hóa các tri thức ngầm định này thành các đặc trưng tường minh (Explicit Features) để làm giàu bộ dữ liệu.
    3. **[ĐÁNH GIÁ TỔNG THỂ]** Chạy lại toàn bộ bảng Benchmark để quan sát sự thay đổi hiệu năng của hệ thống dưới tác động của dữ liệu mới.
*   **Giai đoạn:** Giai đoạn 6 (Optimization & Refinement) - Vòng lặp cải tiến dựa trên tri thức.
*   **Giả thuyết/Câu hỏi:** Việc chuyển đổi các quy luật phi tuyến thành các đặc trưng tường minh có giúp nâng cao độ tin cậy và hiệu năng tổng thể của các thuật toán học máy không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `models/winner_model.joblib` (LightGBM), `Data/data_final_v2.csv`.
*   **Mã nguồn:** 
    *   `src/08_counterfactual_analysis.py` (Phân tích).
    *   `src/09_systematic_refinement.py` (Làm giàu dữ liệu & Re-benchmark).
*   **Đầu ra:** 
    *   File log này (`Logs/08_Counterfactual_Analysis.md`).
    *   Bảng kết quả hệ thống mới: `Data/modeling_results_refined.csv`.

## 3. Chiến lược thực hiện (Strategy)
1.  **Khám phá (Discovery):** Tìm ngưỡng nhạy cảm cho `age_gap_calc` và `int_corr`.
2.  **Làm giàu (Enrichment):** Bổ sung các biến chỉ báo (Indicator variables) vào dữ liệu gốc để hỗ trợ mô hình nhận diện các "vùng quyết định" quan trọng.
3.  **Tái đánh giá khách quan:** Huấn luyện lại tất cả các mô hình trong danh mục đầu tư (Logistic, Tree, Forest, Boosting) trên cùng một điều kiện dữ liệu mới.
4.  **Phân tích sự dịch chuyển:** Quan sát mô hình nào phản ứng tích cực nhất với sự thay đổi để đưa ra kết luận về tính tương thích giữa thuật toán và đặc trưng.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### GĐ 1: Khám phá tri thức (Đã thực hiện)
- [x] **Bước 1: Xác định ngưỡng cắt (Tipping points)** từ kịch bản phản sự thực.

### GĐ 2: Tối ưu hóa hệ thống (Đã thực hiện)
- [x] **Bước 2: Làm giàu bộ đặc trưng (Feature Enrichment)** dựa trên ngưỡng đã tìm thấy.
- [x] **Bước 3: Benchmark toàn diện.** So sánh hiệu năng của toàn bộ các thuật toán.
- [x] **Bước 4: Tổng hợp và Kết luận.** Đánh giá tính hiệu quả của chiến lược làm giàu đặc trưng đối với từng nhóm thuật toán.

## 5. Nhật ký thực thi (Execution Log)
*   **04/06/2026**: Thực hiện Phân tích Phản sự thực trên mô hình LightGBM. Phát hiện các ngưỡng quan trọng: Age Gap (~1.5) và Interest Corr (~0.25).
*   **04/06/2026**: Triển khai Feature Enrichment với 3 đặc trưng mới: `is_age_match`, `is_interest_match`, `match_synergy`.
*   **04/06/2026**: [NÂNG CẤP CÔNG BẰNG] Chạy lại Benchmark toàn hệ thống sử dụng GridSearchCV cho TẤT CẢ các mô hình để đảm bảo so sánh "Táo với Táo" so với Giai đoạn 4.

## 6. Kết quả & Kiểm chứng (Validation)
*   **Bảng Benchmark đối chứng (Validation F1):**

| Mô hình | Trước Enrichment (GĐ 4) | Sau Enrichment (GĐ 6) | Biến động |
| :--- | :---: | :---: | :---: |
| **LightGBM** | **0.4134** | **0.3945** | 📉 -0.0189 |
| **CatBoost** | 0.3837 | 0.3942 | 📈 +0.0105 |
| **XGBoost** | 0.3720 | 0.3840 | 📈 +0.0120 |
| **Random Forest** | 0.3815 | 0.3827 | 📈 +0.0012 |
| **Logistic Reg.** | 0.3256 | 0.3254 | ➖ (Ổn định) |

*   **Kết luận trung lập:** Việc làm giàu dữ liệu không tạo ra một "phép màu" giúp mô hình đơn giản vượt mặt mô hình phức tạp, nhưng nó đã giúp các thuật toán Boosting (CatBoost, XGBoost) cải thiện hiệu năng và tiến sát tới LightGBM. Hệ thống hiện tại có sự đồng thuận cao giữa các thuật toán hàng đầu.

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   **Hiện tượng hội tụ:** Dữ liệu tường minh giúp thu hẹp khoảng cách sai số giữa các thuật toán mạnh. Điều này làm tăng độ tin cậy của dự báo (khi nhiều mô hình cùng cho ra một kết quả tương đương).
*   **Giới hạn của tuyến tính:** Logistic Regression vẫn giữ nguyên điểm số, chứng minh rằng dù có thêm biến chỉ báo, bản chất bài toán Speed Dating vẫn chứa đựng những tương tác phi tuyến cực kỳ phức tạp mà chỉ Tree-based model mới xử lý được.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Cập nhật:** Ghi lại hiện tượng "Sự dịch chuyển của Điểm tối ưu" và tính tương thích giữa Dữ liệu - Thuật toán vào `Logs/Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   Hoàn thiện báo cáo cuối cùng với cái nhìn toàn diện về cả Dữ liệu và Mô hình.


