# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng quy trình hướng cấu hình (Config-driven) để tiền xử lý dữ liệu, đảm bảo tính khách quan tuyệt đối.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Chiến lược Tương hợp Đa tầng (Multi-level Compatibility Distillation):** 
    1.  **Hobby Similarity:** Tính toán khoảng cách sở thích (`abs diff`) để tìm sự đồng điệu.
    2.  **Mutual Surplus (2 Tầng):** Chuyển từ trị tuyệt đối sang phép trừ song phương cho cả 2 phía Subject & Partner, xét trên cả hai lăng kính:
        *   **Tầng 1 (Khiêm tốn - 5_1):** Tôi nghĩ xã hội thấy tôi/họ thế nào vs. Kỳ vọng.
        *   **Tầng 2 (Cái tôi - 3_1):** Tôi tự thấy mình/họ thế nào vs. Kỳ vọng.
    3.  **Anti-Leakage (Chống rò rỉ):** Hợp nhất các bản ghi đối xứng để biến bài toán từ "Dự báo Cá nhân" thành "Dự báo Cặp đôi".
    4.  **Absolute Objectivity (Khách quan tuyệt đối):** Không can thiệp (hard-code) các ngưỡng cắt nhân tạo, để mô hình tự học.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`) với ~155+ đặc trưng và ~4,100 cặp đôi duy nhất.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Dữ liệu quyết định Công cụ"** và **"Hợp nhất nhận thức"**:
1.  **Granular Features:** Giữ nguyên 17 sở thích rời rạc của cả hai bên.
2.  **Hobby Gaps (17 biến):** Độ lệch tuyệt đối về lối sống.
3.  **Mutual Expectation Surplus (24 biến):** 
    *   `Surplus_51 = 5_1 - 1_1_o` (và ngược lại cho S/P): Thặng dư trên góc nhìn thực tế xã hội.
    *   `Surplus_31 = 3_1 - 1_1_o` (và ngược lại cho S/P): Thặng dư trên góc nhìn cái tôi cá nhân.
4.  **Thanh lọc Thực thể (Entity Deletion & Referential Integrity):** Kiên quyết từ chối phương pháp điền khuyết (Imputation) cho những người dùng trống hoàn toàn thông tin. Việc tạo ra các "người nhân bản vô hồn" mang giá trị trung bình sẽ phá hỏng ý nghĩa của các phép tính Surplus. Bất kỳ "bóng ma" nào bị xóa sẽ kéo theo việc xóa bỏ toàn bộ tương tác liên quan của họ.
5.  **Reciprocal Dyadic Deduplication (Loại bỏ Đối xứng):** Mỗi cuộc hẹn A-B tạo ra 2 bản ghi (A đánh B, B đánh A). Do các biến Surplus đã là dạng song phương, 2 bản ghi này chứa lượng thông tin phản chiếu giống hệt nhau. Phải loại bỏ 1 bản ghi đại diện (bằng `pair_id`) để tránh rò rỉ dữ liệu chéo (Cross-leakage) khi chia tập Train/Test.
6.  **Xóa bỏ Tipping Points (Chống định kiến):** Loại bỏ hoàn toàn việc "nhắc bài" mô hình bằng các biến nhị phân tự tạo (như `is_age_match`). Ép mô hình XGBoost phải tự khai phá các ngưỡng này.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 4: Extreme Compatibility Refinement (Đã hoàn thành)
- [x] Tích hợp 17 biến Hobby Gap (Đồng điệu).
- [x] Triển khai bộ 24 biến Mutual Surplus (S và P) cho cả hai tầng nhận thức 3_1 và 5_1.
- [ ] **[KIẾN TRÚC MỚI]** Ngừng xóa bản ghi đối xứng (Deduplication). Thay vào đó, xuất khẩu trực tiếp cột `pair_id` ra file `data_final_v2.csv` để phục vụ cho thuật toán `GroupShuffleSplit` ở Giai đoạn 4.
- [x] Loại bỏ các "bóng ma" và duy trì tính toàn vẹn tham chiếu.
- [x] **[OBJECTIVITY]** Loại bỏ vĩnh viễn các biến Tipping Point để bảo vệ tính khách quan.

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 4: Ultimate Cognitive Data Prep
*   *Mục tiêu: Bắt trọn mọi sắc thái tương hợp từ 'Sự thật khách quan' đến 'Vênh nhận thức cá nhân', đồng thời đảm bảo tính trung thực 100% của tập Test.*

#### **Bước A: Làm sạch hồ sơ người dùng (User Profiles)**
- **A1:** Trích xuất hồ sơ cá nhân. **[ENTITY DELETION]** Phát hiện và xóa sổ vĩnh viễn 7 người dùng (bóng ma) khuyết $\ge$ 20 thông tin hồ sơ.
- **A2:** Điền khuyết bằng median cho các user hợp lệ và thực hiện IQR Clip.

#### **Bước B: Xây dựng hồ sơ cặp đôi (Pair Profiles)**
- **B1:** **[REFERENTIAL INTEGRITY]** Xóa bỏ toàn bộ các cuộc hẹn (interactions) mà Subject hoặc Partner là "bóng ma" đã bị loại ở Bước A1. Sau đó tiến hành Join hồ sơ.
- **B2:** **[CHỐNG RÒ RỈ]** Tạo `pair_id = min(iid, pid)_max(iid, pid)`. Drop duplicates dựa trên `pair_id`, loại bỏ các bản ghi đối xứng. Dữ liệu giảm từ ~8,200 xuống ~4,100 bản ghi cặp đôi duy nhất.

#### **Bước C: Trích xuất Tính tương hợp Cực hạn (Ultimate Compatibility)**
- **C1:** Tính 17 biến `hobby_gap`.
- **C2:** Tính 12 biến Mutual Surplus dựa trên lăng kính khiêm tốn (`5_1`).
- **C3:** Tính 12 biến Mutual Surplus dựa trên lăng kính cái tôi (`3_1`).
- **C4:** **[NO HUMAN BIAS]** Hủy bỏ việc tạo các biến Tipping Points nhân tạo để XGBoost tự học trên dữ liệu liên tục.

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu sau Join cơ bản | ~8,210 hàng |
| **Dữ liệu sau Deduplication (Chỉ giữ Pair)** | **~4,000+ hàng × ~159 cột** |
| Giá trị thiếu | 0 ✓ |
| Trạng thái | Hoàn toàn miễn nhiễm với Reciprocal Leakage ✓ |

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Insight:** Việc từ bỏ các biến Tipping Points tự code (như `is_age_match = age_gap < 1.92`) là một minh chứng cho sự tự tin vào thuật toán. "Hãy để dữ liệu tự lên tiếng" là nguyên tắc cao nhất của khoa học dữ liệu.

## 9. Bước tiếp theo
*   Thực thi đợt huấn luyện cuối cùng trên bộ dữ liệu cặp đôi (Pair-level) trung thực nhất.
