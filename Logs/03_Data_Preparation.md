# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng quy trình hướng cấu hình (Config-driven) để tiền xử lý dữ liệu, đảm bảo tính tái lập.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Chiến lược Tương hợp Đa tầng (Multi-level Compatibility Distillation):** 
    1.  **Hobby Similarity:** Tính toán khoảng cách sở thích (`abs diff`) để tìm sự đồng điệu.
    2.  **Mutual Surplus (2 Tầng):** Chuyển từ trị tuyệt đối sang phép trừ song phương cho cả 2 phía Subject & Partner, xét trên cả hai lăng kính:
        *   **Tầng 1 (Khiêm tốn - 5_1):** Tôi nghĩ xã hội thấy tôi/họ thế nào vs. Kỳ vọng.
        *   **Tầng 2 (Cái tôi - 3_1):** Tôi tự thấy mình/họ thế nào vs. Kỳ vọng.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`) với ~165+ đặc trưng.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Dữ liệu quyết định Công cụ"** và **"Hợp nhất nhận thức"**:
1.  **Granular Features:** Giữ nguyên 17 sở thích rời rạc của cả hai bên.
2.  **Hobby Gaps (17 biến):** Độ lệch tuyệt đối về lối sống.
3.  **Mutual Expectation Surplus (24 biến):** 
    *   `Surplus_51 = 5_1 - 1_1_o` (và ngược lại cho S/P): Thặng dư trên góc nhìn thực tế xã hội.
    *   `Surplus_31 = 3_1 - 1_1_o` (và ngược lại cho S/P): Thặng dư trên góc nhìn cái tôi cá nhân.
4.  **Refined Tipping Points:** Tinh lọc dựa trên ngưỡng nhạy cảm AI thực tế (Age Gap 0.5, Interest Corr 0.3).

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 4: Extreme Compatibility Refinement (Đã hoàn thành)
- [x] Tích hợp 17 biến Hobby Gap (Đồng điệu).
- [x] **[ULTIMATE]** Triển khai bộ 24 biến Mutual Surplus (S và P) cho cả hai tầng nhận thức 3_1 và 5_1.
- [x] Cập nhật Tipping Point nhạy cảm (Age 0.5, Interest 0.3).
- [x] Chuẩn hóa toàn bộ Pipeline.

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 4: Ultimate Cognitive Data Prep
*   *Mục tiêu: Bắt trọn mọi sắc thái tương hợp từ 'Sự thật khách quan' đến 'Vênh nhận thức cá nhân'.*

#### **Bước A: Làm sạch hồ sơ người dùng (User Profiles)**
- **A1:** Trích xuất hồ sơ cá nhân. Xóa 7 người dùng quá thiếu thông tin.
- **A2:** Điền khuyết bằng median và thực hiện IQR Clip.

#### **Bước B: Xây dựng hồ sơ cặp đôi (Pair Profiles)**
- **B1:** Join bảng User Profiles vào bảng Interactions.

#### **Bước C: Trích xuất Tính tương hợp Cực hạn (Ultimate Compatibility)**
- **C1:** Tính 17 biến `hobby_gap`.
- **C2:** **[BỨT PHÁ]** Tính 12 biến Mutual Surplus dựa trên lăng kính khiêm tốn (`5_1`).
- **C3:** **[BỨT PHÁ]** Tính 12 biến Mutual Surplus dựa trên lăng kính cái tôi (`3_1`).
- **C4:** Bổ sung 3 biến Tipping Point nhạy cảm (Age 0.5, Interest 0.3).
- **Kết quả:** Dataset đạt **8,210 hàng x ~167 cột**.

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu sau Join cơ bản | 8,210 hàng × 119 cột |
| **Dữ liệu sau Ultimate Enrichment** | **8,210 hàng × ~167 cột** |
| Giá trị thiếu | 0 ✓ |
| Phân phối mục tiêu | 0: 83.56% / 1: 16.44% |

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Insight:** Việc kết hợp cả 'Cái tôi' (3_1) và 'Thực tế xã hội' (5_1) giúp AI nhận diện được sự tự tin lành mạnh và sự ngộ nhận, từ đó dự báo Match chính xác hơn.

## 9. Bước tiếp theo
*   Thực thi đợt huấn luyện cuối cùng trên bộ dữ liệu giàu tri thức nhất.
