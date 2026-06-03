# Task 04: Huấn luyện mô hình & Xử lý mất cân bằng lớp nâng cao

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** 
    1. Áp dụng các kỹ thuật can thiệp dữ liệu (SMOTE) và tối ưu hóa ngưỡng (Threshold Tuning) để tối đa hóa F1-score trên lớp thiểu số (match=1).
    2. **[MỚI]** Xây dựng Benchmark toàn diện gồm 6 mô hình: **Logistic → Tree → Forest → XGBoost → LightGBM → CatBoost**.
    3. Đánh giá khách quan độ quan trọng của 50 đặc trưng "Pre-match", đặc biệt là kiểm chứng giá trị của các nhóm survey Time 1 (1_1 đến 5_1).
*   **Giai đoạn:** Giai đoạn 4 (Modeling) - Nâng cao theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    * Việc chuyển từ xử lý mất cân bằng "thụ động" sang "chủ động" sẽ cải thiện F1-score lớp Match lên trên mức 0.5?
    * "Bộ ba vương quyền" Boosting (XGB/LGB/CAT) có vượt trội hơn Random Forest truyền thống trên bộ dữ liệu này không?
    * Trong 5 nhóm survey Time 1, nhóm nào thực sự có sức mạnh dự báo (Predictive power) cao nhất?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/data_final_v2.csv` (50 đặc trưng Pre-match).
*   **Mã nguồn:** `src/04_modeling.py` (Phiên bản Refactored).
*   **Đầu ra:** 
    *   File log này (`Logs/04_Modeling.md`).
    *   Bảng so sánh hiệu năng của 6 mô hình.
    *   **Biểu đồ Feature Importance** (sử dụng mô hình Boosting tốt nhất).

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý: **"Cân bằng khi học, Khách quan khi thi"** và **"Trung thực trong kiểm định"**.
1.  **Honest Sequence (Thứ tự trung thực):** Chia dữ liệu TRƯỚC KHI thực hiện SMOTE.
2.  **Realistic Validation (Kiểm định thực tế):** Tập Validation và Test KHÔNG chứa dữ liệu ảo.
3.  **Comprehensive Benchmark:** Tích hợp bộ ba Boosting mạnh nhất hiện nay (XGBoost, LightGBM, CatBoost) với các cơ chế xử lý imbalance chuyên biệt (`is_unbalance`, `auto_class_weights`).
4.  **Class-Specific Focus:** Sử dụng `scoring='f1'` (Binary Class 1) trong GridSearchCV.
5.  **Threshold Optimization:** Tìm ngưỡng $T$ tối ưu trên tập **Validation thực tế** để tối đa hóa F1-score cho lớp 1.
6.  **Feature Importance Ranking:** "Xét xử" 50 biến đầu vào bằng các thuật toán Boosting để loại bỏ định kiến chủ quan.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [ ] **Bước 1: Chia tách dữ liệu (60/20/20)**
    *   Chia `X_train, X_val, X_test` với tham số `stratify=y`.
- [ ] **Bước 2: SMOTE trên tập Train**
    *   Chỉ áp dụng SMOTE cho `X_train, y_train`.
- [ ] **Bước 3: Cấu hình Benchmark 6 mô hình**
    *   Tích hợp LightGBM (`is_unbalance=True`).
    *   Tích hợp CatBoost (`auto_class_weights='Balanced'`).
- [ ] **Bước 4: Threshold Optimizer**
    *   Tìm ngưỡng tối ưu trên tập Val thực tế cho toàn bộ 6 mô hình.
- [ ] **Bước 5: Phân tích Feature Importance**
    *   Trích xuất và vẽ biểu đồ Top 30 đặc trưng.
    *   Đối chiếu sức mạnh của 5 nhóm survey Time 1.

## 5. Nhật ký thực thi (Execution Log)

### 🔄 Đang chờ thực hiện (Phase 2: 6-Model Benchmark & Feature Sentencing)
*   *Ghi chú: Phase 1 (Baseline) đạt F1 ~0.32 với Random Forest. Cần kiểm tra xem LightGBM và CatBoost có đẩy được con số này lên không.*

## 6. Kết quả & Kiểm chứng (Validation)
*   *(Sẽ cập nhật sau khi chạy code Phase 2)*

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)
*   *(Sẽ cập nhật sau khi chạy code Phase 2)*

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **⚠️ Yêu cầu:** Cập nhật hiệu quả của SMOTE vs Class Weight vào `Logs/Reflection_and_Knowledge_Base.md`.

## 9. Bước tiếp theo
*   Triển khai code logic cho SMOTE và Threshold Tuning trong `src/04_modeling.py`.
