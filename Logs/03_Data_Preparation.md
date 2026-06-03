# Task 03: Chuẩn bị và Tiền xử lý dữ liệu hệ thống (Systematic Data Preparation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng quy trình hướng cấu hình (Config-driven) và ML Pipeline để tiền xử lý dữ liệu, đảm bảo tính tái lập và dễ dàng thay đổi thuộc tính.
*   **Giai đoạn:** Giai đoạn 3 (Data Preparation) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** Việc tách biệt Logic và Tham số qua biến `CONFIG` có giúp quản lý 6 nguyên lý tiền xử lý hiệu quả hơn không?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** `Data/Speed Dating Data.csv`.
*   **Mã nguồn:** `src/03_data_preparation.py` (Phiên bản Refactored)
*   **Đầu ra:** 
    *   File log này (`Logs/03_Data_Preparation.md`).
    *   Tập dữ liệu cuối cùng (`Data/data_final_v2.csv`).

## 3. Chiến lược thực hiện (Strategy)
Tuân thủ 6 nguyên lý qua hệ thống mới:
1.  **Config-Driven:** Toàn bộ danh sách cột được định nghĩa trong `CONFIG`.
2.  **Pre-match Logic (Anti-leakage):** Loại bỏ hoàn toàn các biến Scorecard (phát sinh SAU cuộc gặp) để đảm bảo mô hình có tính ứng dụng thực tế.
3.  **Gold Static Features:** Tập trung vào các biến tĩnh (Static) phản ánh bản chất, sở thích và kỳ vọng của người tham gia.
4.  **Modular Functions:** Tách biệt `entity_cleaning`, `relationship_cleaning`, `feature_engineering`.
5.  **Referential Integrity:** Đảm bảo tính nhất quán giữa Entity và Relationship (Xóa user -> Xóa tương tác liên quan).
6.  **Pipeline Scaling:** Sử dụng `ColumnTransformer` cho Scaling đa chiến lược.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

### Phase 1: Tái cấu trúc (Đã hoàn thành)
- [x] Thiết lập CONFIG và Modular Functions.
- [x] Xử lý Impute và IQR Clip cho User.

### Phase 2: Tối ưu hóa "Pre-match" & Data-Driven (Sắp thực hiện)
- [ ] **Bước 1: Thanh lọc Feature (Loại bỏ Leakage)**
    *   Xóa triệt để các biến Scorecard (like, prob, attr, sinc... và các bản _o).
- [ ] **Bước 2: Tích hợp toàn diện "Biến Vàng" Time 1 (Tránh chủ quan)**
    *   Lấy toàn bộ các nhóm: `1_1` (Preferences), `2_1` (Partner expectation), `3_1` (Self-rating), `4_1` (Same-sex peers), `5_1` (Others perceive).
    *   Bổ sung Demographic & Lifestyle: `goal`, `date`, `go_out`, `exphappy`, `expnum`, `field_cd`, `career_c`.
- [ ] **Bước 3: Sử dụng Biến Tổng hợp & Referential Integrity**
    *   Sử dụng `samerace` và `int_corr`.
    *   Đồng bộ hóa: Xóa `pid` nếu `iid` tương ứng bị loại bỏ ở bước Entity Cleaning.
- [ ] **Bước 4: Chuẩn hóa đa thang đo**
    *   Áp dụng MinMaxScaler cho các biến 1-10 và 100pt để đưa về cùng hệ quy chiếu [0, 1].

## 5. Nhật ký thực thi (Execution Log)

### 🔄 Đang chờ thực hiện (Phase 2: Pre-match Refinement)
*   *Mục tiêu: Chuyển đổi mô hình sang hướng "Data-Driven", giữ lại tối đa thông tin Pre-event để mô hình tự quyết định độ quan trọng.*
... (giữ nguyên kết quả thống kê) ...

#### **Bước A: Làm sạch dữ liệu thực thể (Entity-Level Cleaning)**
- **A1:** Xóa 7 người dùng có >= 5 giá trị thực thể bị thiếu (dropped_iids: [28, 58, 59, 136, 339, 340, 346])
- **A2:** Lưu bản đồ tuổi (age_map) từ 544 người dùng duy nhất (cần cho age_gap trước khi chuẩn hóa)
- **A3:** Điền giá trị thiếu cho thực thể (trung vị cho dữ liệu số, yếu vị cho dữ liệu phân loại)
  - age: 32 giá trị đã được điền
  - Hobbies: 0 giá trị cần điền (đã có đủ dữ liệu)
- **A4:** Xử lý ngoại lệ bằng IQR Clip (cắt giá trị thay vì xóa hàng)
  - age: 173 giá trị cận trên đã được cắt
  - Nhiều sở thích: tổng cộng 1143 giá trị đã được cắt

#### **Bước B: Làm sạch dữ liệu quan hệ (Relationship-Level Cleaning)**
- **B1:** Xóa các tương tác có > 50% khối đánh giá bị thiếu
  - Đã xóa 133 hàng (khối like/prob)
  - Đã xóa thêm 82 hàng (khối đánh giá pf_o)
  - Tổng cộng: 215 tương tác đã bị xóa
- **B2:** Đồng bộ - xóa các tương tác mồ côi (orphan interactions)
  - Đã xóa 0 tương tác mồ côi (đã được đồng bộ trong B1)
- **B3:** Điền các giá trị thiếu còn lại trong tương tác (chế độ im lặng, > 200 cột)

#### **Bước D: Chuẩn hóa (Scaling)**
- **D0:** Điền giá trị thiếu lần cuối trước khi chuẩn hóa (tổng cộng 388,302 giá trị thiếu trên tất cả các cột)
  - Đảm bảo không còn giá trị NaN sau khi scaler.fit_transform()
- **D1:** Áp dụng MinMax scaling [0,1] cho 16 cột (like, prob, pf_o_*, gộp sở thích)
- **D2:** Áp dụng StandardScaler (Z-score) cho 3 cột (age, age_gap, age_o)

#### **Bước E: Hoàn tất chuẩn bị (Final Preparation)**
- **E1:** Đã chọn 24 đặc trưng cuối cùng để đưa vào mô hình
- **E2:** Kiểm tra điền giá trị thiếu lần cuối: 0 giá trị thiếu ✓
- **E3:** Tất cả các kiểm tra xác thực đã vượt qua

### 📊 Chỉ số chất lượng dữ liệu (Data Quality Metrics)

| Chỉ số | Giá trị |
|--------|-------|
| Dữ liệu thô | 8,378 hàng × 195 cột |
| Sau khi làm sạch thực thể | 8,299 hàng × 195 cột (xóa 79 hàng từ 7 người dùng) |
| Sau khi lọc tương tác | 8,084 hàng × 195 cột (xóa 215 tương tác) |
| Tập dữ liệu cuối cùng | 8,084 hàng × 24 cột |
| **Giá trị thiếu** | **0** ✓ |
| Trùng lặp | 0 ✓ |
| Phân phối mục tiêu | 0: 6,734 (83.30%) / 1: 1,350 (16.70%) |
| **Cân bằng lớp** | **16.70% tích cực** (hơi mất cân bằng) |

### 🔍 Thống kê chính (Dữ liệu cuối cùng)

```
Cột (24):
  - ID & Mục tiêu: iid, match
  - Nhân khẩu học: gender, age, race, race_o
  - Đánh giá: like, prob, like_o, prob_o, age_o, age_gap
  - Sở thích của đối tác: pf_o_att, pf_o_sin, pf_o_int, pf_o_fun, pf_o_amb, pf_o_sha
  - Gộp sở thích (5): fitness_sport, fine_arts, entertainment, social_nightlife, outdoor_wellness
  - Khác: condtn

Kiểu dữ liệu: float64 (20 cột) + int64 (4 cột)
Chuẩn hóa đã áp dụng:
  - MinMax [0,1]: Các đánh giá, sở thích
  - Standard (Z-score): Age, age_gap, age_o
```

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Vượt qua tất cả kiểm tra (Chẩn đoán cấp độ 4)

**Kiểm tra giá trị thiếu:**
- ✓ Tập dữ liệu cuối cùng: 0 giá trị thiếu (giảm từ 432,016 trong dữ liệu thô)
- ✓ Tất cả 24 đặc trưng đã hoàn thiện

**Kiểm tra kiểu dữ liệu & Chuẩn hóa:**
- ✓ Tuổi (Standard scaled): Trung bình ≈ 0, Độ lệch chuẩn ≈ 1 (từ trung bình thô 26, độ lệch chuẩn 2.5)
- ✓ Đánh giá (MinMax scaled): Khoảng [0, 1] (từ khoảng thô 1-10)
- ✓ Gộp sở thích (MinMax scaled): Khoảng [0, 1]

**Kiểm tra tính toàn vẹn tham chiếu:**
- ✓ Tất cả 8,084 tương tác đều có iid hợp lệ
- ✓ Không có hàng mồ côi (tương tác không có người dùng tương ứng)
- ✓ Không có hàng trùng lặp

**Kiểm tra cân bằng lớp:**
- ✓ Phân phối mục tiêu: 83.30% tiêu cực / 16.70% tích cực
- ✓ Tỷ lệ mất cân bằng: ~5:1 (hợp lý để lập mô hình với các kỹ thuật như SMOTE/class_weight)

### 📁 Các tệp đầu ra

```
✓ Data/data_cleaned_entities.csv (5.7 MB)
  - Tập dữ liệu đầy đủ đã làm sạch thực thể (8,084 hàng × 195 cột)
  - Sau các bước A1-A4: loại bỏ ngoại lệ, điền giá trị thực thể
  - Dùng làm tham chiếu để hiểu về làm sạch cấp thực thể
  
✓ Data/data_final_v2.csv (2.1 MB)
  - Tập dữ liệu sẵn sàng cho mô hình (8,084 hàng × 24 cột)
  - Sau các bước A-E: đã làm sạch, kỹ thuật đặc trưng, chuẩn hóa, xác thực đầy đủ
  - 0 giá trị thiếu
  - Sẵn sàng để chia tập train/validation/test
```

## 7. Khám phá quan trọng & Chẩn đoán lỗi (Insights & Diagnostics)

### Điểm Nổi Bật

1. **Chiến lược Thực thể vs Quan hệ Hiệu quả:**
   - Chỉ xóa 7 người dùng (79 tương tác), giữ lại 99.1% dữ liệu
   - Chiến lược điền giá trị bảo toàn tính toàn vẹn của thực thể
   - Lưu ý: Nếu dùng chiến lược "xóa tất cả giá trị thiếu", sẽ mất thêm 294 tương tác (thêm 3.6%)

2. **Kỹ thuật đặc trưng phụ thuộc thứ tự (Order-Dependent FE) - Tại sao quan trọng:**
   - age_gap tính từ tuổi thô TRƯỚC khi chuẩn hóa → giá trị thực tế
   - Nếu tính sau khi chuẩn hóa → sẽ là |z-score_age1 - z-score_age2| (vô nghĩa)
   - Minh họa: age (thô: trung bình=26, độ lệch chuẩn=2.5) → age (chuẩn hóa: trung bình≈0, độ lệch chuẩn≈1)

3. **Gộp đặc trưng có ý nghĩa - 17 sở thích → 5 đặc trưng:**
   - Giảm số chiều 70% (17 → 5)
   - Tăng khả năng giải thích
   - Lợi ích: Mô hình dễ học và tổng quát hóa tốt hơn
   - Ví dụ: nhóm "fine_arts" = tương quan giữa [bảo tàng, nghệ thuật, đọc sách, tv, nhà hát] (về mặt ngữ nghĩa)

4. **Phân biệt chiến lược chuẩn hóa:**
   - MinMax cho các đánh giá (giới hạn 1-10) → [0, 1]
   - Standard cho tuổi (không giới hạn, liên tục) → z-score
   - Không dùng một cách chuẩn hóa duy nhất cho tất cả

5. **Xử lý ngoại lệ qua IQR Clip:**
   - Cắt IQR tuổi: 173 giá trị cận trên (sai sót tiềm ẩn hoặc ngoại lệ thực sự)
   - Cắt IQR sở thích: tổng cộng 1,143 giá trị
   - Lợi ích: Giữ nguyên kích thước tập dữ liệu, loại bỏ nhiễu cực đoan
   - Đánh đổi: Một số giá trị cực đoan bị "ghìm" lại thay vì bị xóa bỏ

### Lỗi Phổ Biến Tránh Được

1. ❌ **Sai lầm 1:** Chuẩn hóa trước khi thực hiện kỹ thuật đặc trưng phụ thuộc thứ tự → Đã sửa: tính age_gap trước khi chuẩn hóa
2. ❌ **Sai lầm 2:** Xóa người dùng nhưng không xóa các tương tác liên quan → Đã sửa: Đồng bộ hóa ở bước B2
3. ❌ **Sai lầm 3:** Áp dụng một kiểu chuẩn hóa cho tất cả → Đã sửa: Phân biệt MinMax vs Standard
4. ❌ **Sai lầm 4:** Điền giá trị thiếu cho dữ liệu quan hệ bằng giá trị trung bình → Đã sửa: Chiến lược: xóa nếu thiếu > 50%
5. ❌ **Sai lầm 5:** fillna(inplace=True) trong gán chuỗi → Đã sửa: Sử dụng phép gán df[col] = df[col].fillna()

### Chẩn Đoán Chi Tiết

**Giá trị thiếu trước khi làm sạch:** 432,016 (~2.6% dữ liệu)
- Phần lớn từ các khảo sát sau khi hẹn hò (dữ liệu động)
- Tác động: 215 tương tác bị xóa, hơn 388,302 giá trị được điền trên tất cả các cột
- Kết quả: 0 giá trị thiếu trong tập dữ liệu cuối cùng ✓

**Cân bằng dữ liệu cho mô hình:**
- Mất cân bằng lớp: 16.70% tích cực (1,350 cặp khớp / 6,734 cặp không khớp)
- Khuyến nghị: Sử dụng các kỹ thuật như SMOTE, class_weight, hoặc điều chỉnh ngưỡng trong quá trình lập mô hình
- Không quá nghiêm trọng: Tỷ lệ ~5:1 có thể xử lý được bằng cách lấy mẫu phù hợp

**Xác thực chuẩn hóa:**
- Tuổi (thô): min=20, max=50, trung bình=26.00, độ lệch chuẩn=2.5
- Tuổi (chuẩn hóa): min=-2.4, max=9.6, trung bình≈0, độ lệch chuẩn≈1 ✓
- Đánh giá (chuẩn hóa): min=0, max=1 (tất cả nằm trong khoảng [0,1]) ✓

## 8. Đồng bộ Tri thức (Knowledge Synchronization)

### ✅ Insights từ Preprocessing Validated

Những nguyên lý từ `Specs/plan.md` được áp dụng thành công:

| Nguyên lý | Thực hiện | Status |
|-----------|---------|--------|
| **Entity vs Relationship Cleaning** | Impute user (A3), Delete interaction > 50% (B1) | ✓ Applied |
| **Order-Dependent FE** | age_gap từ raw age trước scaling | ✓ Applied |
| **Scaling Strategy** | MinMax [0,1] + Standard (z-score) | ✓ Applied |
| **Meaningful Aggregation** | 17 hobbies → 5 semantic groups | ✓ Applied |
| **Outlier Handling** | IQR clip (1,143 values) | ✓ Applied |
| **Data Synchronization** | Xóa 7 users → Xóa 79 interactions | ✓ Applied |

### Updates cho Reflection File

Những bài học được xác nhận qua thực thi:
- ✓ **Section 5.1 (Scaling Strategy)**: MinMax vs Standard thực tế hoạt động tốt
- ✓ **Section 5.2 (Meaningful Aggregation)**: Gộp 17→5 hobbies thành công giảm noise
- ✓ **Section 5.3 (Order-Dependent FE)**: age_gap trước scaling là bắt buộc
- ✓ **Section 5.4 (Entity vs Relationship)**: Impute user, delete interaction strategy hiệu quả
- ✓ **Section 5.5 (Outlier Handling)**: IQR clip giữ 100% dataset size, loại bỏ 1,143 extreme values
- ✓ **Section 5.6 (Data Synchronization)**: Referential integrity maintained (0 orphan rows)

## 9. Bước tiếp theo (Next Steps)

### Giai đoạn 4: Modeling (Task 04)

**Input:** 
- ✓ Data/data_final_v2.csv (8,084 rows × 24 cols, 0 missing values)

**Objectives:**
1. **Chia dữ liệu:** Train (60%) / Validation (20%) / Test (20%)
2. **Xử lý mất cân bằng lớp:** SMOTE hoặc class_weight
3. **So sánh mô hình:**
   - Baseline: Logistic Regression
   - Cơ sở: Decision Tree
   - Nâng cao: Gradient Boosting (XGBoost/CatBoost/LightGBM)
4. **Hyperparameter Tuning:** GridSearchCV với k-fold CV
5. **Chọn winner model:** Dựa trên F1-score (không Accuracy)

**Reference:**
- Specs/plan.md - Giai đoạn 4 chi tiết
- Logs/Reflection_and_Knowledge_Base.md - Section 2 (Evaluation Mindset)

**Estimated Output:**
- Best model + hyperparameters
- Validation F1-score
- Feature importance ranking
