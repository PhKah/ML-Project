# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Chuyển đổi sang quy trình **Trích xuất Dữ liệu Thô Tinh khiết** (Raw Feature Extraction) nhằm triệt tiêu hoàn toàn rò rỉ dữ liệu hệ thống (Global Data Leakage).
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) - Phiên bản Chuẩn hóa Kiến trúc.
*   **Chiến lược Tương hợp Đa tầng (Multi-level Compatibility Distillation):** 
    1.  **Hobby Similarity:** Tính toán khoảng cách sở thích (`abs diff`) thô.
    2.  **Feature Consolidation (Noise Reduction):** [MỚI] Gộp 17 đặc trưng sở thích nhỏ lẻ thành 1 chỉ số tổng hòa (`total_hobby_gap`) nhằm tránh hiện tượng Overfitting do cây quyết định phân nhánh quá vụn vặt.
    3.  **Mutual Surplus (2 Tầng):** Tính toán thặng dư kỳ vọng (phép trừ có dấu) trên lăng kính 3_1 và 5_1 ở dạng số đo nguyên bản.
    4.  **Anti-Leakage (Strict Isolation):** Loại bỏ hoàn toàn các bước tính toán thống kê toàn cục (Mean, Std, Median) ở pha này.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`) chứa ~160 đặc trưng **Dạng Thô**.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Cách ly Thống kê"** (Statistical Isolation) và **"Cô đọng Tín hiệu"** (Signal Condensation):
1.  **Pure Data Extraction:** Chỉ thực hiện các biến đổi mang tính logic (Join, Diff, Surplus) trên từng dòng dữ liệu.
2.  **Aggregation over Granularity:** Ưu tiên sử dụng các chỉ số tổng hợp cho những đặc trưng có độ biến thiên cao và giá trị lẻ (như sở thích cá nhân) để mô hình tập trung vào "Vibe" tổng thể thay vì các tiểu tiết gây nhiễu.
3.  **No Global Imputation:** Kiên quyết không điền khuyết bằng Trung vị toàn cục tại đây.
4.  **No Pre-Scaling:** Việc lựa chọn chuẩn hóa sẽ được thực hiện trong Pipeline của `src/04`.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 5: Pipeline Overhaul - Leakage Eradication (Đã hoàn thành)
- [x] **[DELETION]** Gỡ bỏ hàm `apply_scaling` và các bộ biến đổi tỷ lệ toàn cục.
- [x] **[DELETION]** Gỡ bỏ lệnh `fillna(median)` toàn cục cho tập Interaction.
- [x] Duy trì bộ 24 biến Mutual Surplus (S và P) ở dạng thô.
- [x] Duy trì cột `pair_id` cho Group Split.

### Phase 6: Signal Condensation - Overfitting Prevention (Đang thực hiện)
- [ ] **[NEW]** Triển khai hàm gộp 17 biến `_gap` sở thích thành biến `total_hobby_gap` (tổng hoặc trung bình).
- [ ] Loại bỏ các biến `gap` sở thích lẻ để làm "sạch" không gian đặc trưng.
- [x] Đảm bảo logic xóa Bóng ma và tính toàn vẹn tham chiếu.

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 5: Raw Data Standardization
*   *Mục tiêu: Đưa Pipeline về chuẩn khoa học dữ liệu, biến Giai đoạn 3 thành 'Data Factory' cung cấp nguyên liệu thô sạch.*

### 🔄 Khởi động Phase 6: Feature Aggregation
*   *Lý do:* Quan sát thấy 7/10 đặc trưng quan trọng nhất của mô hình là các biến Gap sở thích lẻ (concerts, yoga...), dẫn đến rủi ro Overfitting cực cao. Việc gộp nhóm sẽ giúp mô hình "thông minh" và "tổng quát" hơn.

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Triết lý Signal-to-Noise Ratio:** Trong dữ liệu hành vi, việc có quá nhiều biến yếu (Weak predictors) sẽ dìm chết các biến mạnh (Strong predictors). Việc gộp nhóm đặc trưng là một bước "phẫu thuật" cần thiết để tăng tỷ lệ Tín hiệu/Nhiễu, giúp mô hình đạt được Precision thực tế thay vì Accuracy ảo.

## 9. Bước tiếp theo
*   Cập nhật `src/03_data_preparation.py` để thực hiện gộp nhóm sở thích.
*   Chạy lại `src/04_modeling.py` để kiểm chứng hiệu năng trên bộ đặc trưng tinh hoa mới.
