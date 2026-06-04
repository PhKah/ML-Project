# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng quy trình hướng cấu hình (Config-driven) để tiền xử lý dữ liệu, đảm bảo tính tái lập.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Chiến lược Tương hợp (Compatibility Distillation):** Ngoài việc giữ nguyên bản chất khách quan, chúng ta thực hiện "chiết xuất tri thức" bằng cách tính toán trực tiếp các **Biến chênh lệch (Gaps)** và các **Chỉ báo nhạy cảm (Sensitive Indicators)** tìm được từ mô hình.

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py`.
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`) với 146 đặc trưng.

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Dữ liệu quyết định Công cụ"** và **"Số hóa sự tương hợp"**:
1.  **Granular Features:** Giữ nguyên 17 sở thích rời rạc của cả hai bên.
2.  **Compatibility Gaps:** Bổ sung 17 cột `hobby_gap` và 6 cột `expectation_gap` (abs diff).
3.  **Refined Tipping Points:** Tinh lọc các ngưỡng cắt dựa trên Phân tích Phản sự thực thực tế:
    *   **Age Gap:** Rút ngắn xuống **0.5 năm** (điểm rơi xác suất mạnh nhất).
    *   **Interest Corr:** Nâng lên **0.30** (ngưỡng tin cậy của AI).
4.  **Referential Integrity:** Đảm bảo tính toàn vẹn hồ sơ thực thể.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 4: Compatibility-Enhanced Refinement (Đã hoàn thành)
- [x] Tích hợp 23 biến Gap (Chênh lệch sở thích & kỳ vọng).
- [x] **[REFINED]** Áp dụng 3 biến Tipping Point nhạy cảm (`is_age_match` với 0.5, `is_interest_match` với 0.3).
- [x] Chuẩn hóa toàn bộ Pipeline theo triết lý "Dẫn dắt bởi tri thức".

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 4: Knowledge-Driven Data Prep
*   *Mục tiêu: Chuyển hóa các quy luật phi tuyến nhạy cảm nhất của AI thành đặc trưng tường minh.*

#### **Bước A: Làm sạch hồ sơ người dùng (User Profiles)**
- **A1:** Trích xuất hồ sơ cá nhân. Xóa 7 người dùng quá thiếu thông tin.
- **A2:** Điền khuyết bằng median và thực hiện IQR Clip.

#### **Bước B: Xây dựng hồ sơ cặp đôi (Pair Profiles)**
- **B1:** Join bảng User Profiles vào bảng Interactions.

#### **Bước C: Trích xuất Tính tương hợp (Compatibility Features)**
- **C1:** Tính 17 biến `hobby_gap`.
- **C2:** Tính 6 biến `expectation_gap`.
- **C3:** **[TỐI ƯU CỰC HẠN]** Bổ sung 3 biến Tipping Point nhạy cảm dựa trên biểu đồ kịch bản (Age Gap = 0.5, Interest Corr = 0.3).
- **Kết quả:** Dataset đạt **8,210 hàng x 146 cột**.

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu sau Join cơ bản | 8,210 hàng × 119 cột |
| **Dữ liệu sau Enrichment (Gaps)** | **8,210 hàng × 146 cột** |
| Giá trị thiếu | 0 ✓ |
| Phân phối mục tiêu | 0: 83.56% / 1: 16.44% |

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Insight:** AI nhạy cảm với chênh lệch tuổi tác hơn ta tưởng (ngưỡng 0.5 năm thay vì 2 năm). Việc bám sát "thực tế của AI" giúp dữ liệu trở nên cực kỳ tinh khiết.

## 9. Bước tiếp theo
*   Chạy lại quy trình huấn luyện để xác nhận hiệu năng trên bộ đặc trưng tinh khiết nhất.
