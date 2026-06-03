# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng quy trình hướng cấu hình (Config-driven) và ML Pipeline để tiền xử lý dữ liệu, đảm bảo tính tái lập và dễ dàng thay đổi thuộc tính.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** Việc giữ nguyên bản chất khách quan của dữ liệu (không gộp sở thích, không tự nhân biến so khớp) có giúp mô hình phi tuyến (Boosting) học được các quy luật tốt hơn không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py` (Phiên bản Granular & Objective)
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`).

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ nguyên lý **"Dữ liệu quyết định Công cụ"** và **"Khách quan tối đa"**:
1.  **Granular Features:** Giữ nguyên các biến sở thích rời rạc (ví dụ: `sports` khác `tvsports`) vì chúng phản ánh lối sống khác nhau.
2.  **Raw Dyadic Pair:** Cung cấp đầy đủ cặp biến đối ứng (Subject Pref vs Partner Self-rating) thay vì tự nhân chúng lại với nhau.
3.  **Objective Imputation:** Sử dụng median cho các biến liên tục trên bảng User Profile sạch.
4.  **Referential Integrity:** Đảm bảo mỗi interaction đều có hồ sơ người dùng hợp lệ từ cả hai phía.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 4: Granular & Objective Refinement (Đã hoàn thành)
- [x] Tách rời các nhóm sở thích đã gộp trước đó.
- [x] Loại bỏ các biến `match_` tự tính bằng phép nhân.
- [x] Giữ lại `age_gap_calc` như một đặc trưng gợi ý (shortcut).

## 5. Nhật ký thực thi (Execution Log)

### ✅ Hoàn thành Phase 4: Granular & Objective Data Prep
*   *Mục tiêu: Loại bỏ định kiến chủ quan của người lập trình, để mô hình tự tìm ra mối quan hệ giữa các thuộc tính.*

#### **Bước A: Làm sạch hồ sơ người dùng (User Profiles)**
- **A1:** Trích xuất hồ sơ cá nhân (1 dòng/iid). Xóa 7 người dùng có hồ sơ quá thiếu (> 20 ô).
- **A2:** Không thực hiện gộp sở thích (Aggregation). Giữ nguyên 17 biến sở thích rời rạc.
- **A3:** Điền khuyết bằng median và thực hiện IQR Clip để kiềm chế nhiễu.

#### **Bước B: Xây dựng hồ sơ cặp đôi (Pair Profiles)**
- **B1:** Join bảng User Profiles vào bảng Interactions cho cả `iid` và `pid`.
- **B2:** Referential Integrity: Xóa 10 tương tác mồ côi.
- **Kết quả:** Dataset có **8,210 hàng x 119 cột**.

#### **Bước C: Đặc trưng tương tác khách quan**
- **C1:** Chỉ giữ lại `age_gap_calc` (hiệu số tuổi tuyệt đối).
- **C2:** Dựa vào `int_corr` sẵn có của dữ liệu thô.

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics - Phase 4)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu thô | 8,378 hàng × 195 cột |
| Sau khi làm sạch & Join | 8,210 hàng × 119 cột |
| **Giá trị thiếu** | **0** ✓ |
| Phân phối mục tiêu | 0: 6,860 (83.56%) / 1: 1,350 (16.44%) |

### 🔍 Thống kê chính (Dữ liệu Granular)
```
Cột (119):
  - Subject Profile: age, gender, race, goal, 17 hobbies, 5 prefs, 5 self-ratings...
  - Partner Profile: age_o, race_o, goal_o, 17 hobbies_o, 5 prefs_o, 5 self-ratings_o...
  - Dyadic: int_corr, age_gap_calc.
```

## 8. Đồng bộ Tri thức (Knowledge Synchronization)
*   **Insight:** Việc giữ dữ liệu ở dạng thô (granular) giúp mô hình có nhiều "nguyên liệu" hơn để học các tương tác phức tạp.

## 9. Bước tiếp theo
*   Chuyển sang huấn luyện lại mô hình với bộ dữ liệu 119 cột.
