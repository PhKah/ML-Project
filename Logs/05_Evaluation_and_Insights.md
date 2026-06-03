# Task 05: Đánh giá hệ thống & Chẩn đoán 5 cấp độ (Systematic Evaluation)

## 1. Mục tiêu & Bối cảnh
*   **Mục tiêu:** Áp dụng hệ thống chẩn đoán lỗi 5 cấp độ đồng bộ với quy trình hướng cấu hình, sử dụng mô hình tốt nhất (**CatBoost**) và ngưỡng tối ưu đã tìm được.
*   **Giai đoạn:** Giai đoạn 5 (Evaluation & Insights) theo `plan.md`.
*   **Giả thuyết/Câu hỏi:** 
    * Mô hình sau khi tối ưu ngưỡng có duy trì được tính công bằng (fairness) và sự ổn định trên các phân khúc nhân khẩu học khác nhau không?
    * Đặc trưng "Người khác nghĩ gì về mình" (Nhóm 5) có thực sự giúp giảm sai số FP/FN trong thực tế?

## 2. Đầu vào & Đầu ra (Input/Output)
*   **Đầu vào:** Mô hình tốt nhất (**CatBoost**), ngưỡng tối ưu ($T=0.27$), và tập dữ liệu 50 đặc trưng Pre-match.
*   **Mã nguồn:** `src/05_evaluation.py` (Phiên bản đồng bộ CONFIG).
*   **Đầu ra:** 
    *   File log này (`Logs/05_Evaluation_and_Insights.md`).
    *   Dashboard chẩn đoán 5 cấp độ (`plots/evaluation_diagnostics.png`).
    *   Báo cáo sẵn sàng vận hành (Production Readiness).

## 3. Chiến lược thực hiện (Strategy)
*   **Sync with CONFIG:** Đọc toàn bộ tham số thí nghiệm từ biến `CONFIG` chung của dự án.
*   **Honest Metrics:** Tập trung vào F1-score lớp Match (Class 1) trên dữ liệu chưa bao giờ gặp.
*   **5-Level Framework:**
    1. **Bề nổi:** Metrics tổng thể (Precision, Recall, F1 lớp 1).
    2. **Cấu trúc:** Bias-Variance qua Learning Curves.
    3. **Ẩn sâu:** Phân tích chi tiết FP/FN và sự đánh đổi (Confusion Matrix heatmap).
    4. **Gốc rễ:** Kiểm tra chất lượng dữ liệu của các mẫu lỗi.
    5. **Vận hành:** Kiểm tra tính công bằng theo Giới tính/Chủng tộc.

## 4. Hướng dẫn thực hiện chi tiết (Checklist & Tutorial)

- [x] **Bước 1: Đồng bộ hóa mã nguồn**
    *   Cập nhật `src/05_evaluation.py` để sử dụng `CONFIG`.
- [x] **Bước 2: Áp dụng Ngưỡng tối ưu**
    *   Sử dụng hàm `predict_proba` và ngưỡng $T=0.27$ từ CatBoost.
- [x] **Bước 3: Chạy Chẩn đoán 5 cấp độ**
    *   Thực hiện tuần tự các phân tích từ Level 1 đến Level 5.
- [x] **Bước 4: Tổng kết Insights**
    *   Đối chiếu kết quả với các giả thuyết ban đầu (H1, H2, H3).

## 5. Nhật ký thực thi (Execution Log)

### ✅ Thực thi thành công (Date: 2026-06-03)

#### **CẤP ĐỘ 1: Chỉ số bề nổi (Basic Performance)**

Hiệu năng tập Test với mô hình tốt nhất (CatBoost) tại ngưỡng $T=0.27$:

| Chỉ số | Giá trị |
|--------|-------|
| **F1-score (Class 1)** | **0.32** ✓ |
| **Precision (Class 1)** | 0.25 |
| **Recall (Class 1)** | 0.46 |
| **ROC-AUC** | 0.6446 |

**Giải thích:**
- ⚠️ Hiệu năng thực tế phản ánh đúng độ khó của bài toán "Pre-match".
- ✓ Recall (0.46) cho thấy mô hình bắt được gần một nửa số cặp match tiềm năng.
- ⚠️ Precision (0.25) cho thấy tỷ lệ báo động giả còn cao (1 đúng / 3 sai).

#### **CẤP ĐỘ 2: Cấu trúc (Bias-Variance Diagnostic)**

**Phân tích đường cong học tập (Learning Curve):**
```
Khoảng cách (Train F1 - Val F1): 0.0870

Chẩn đoán: ⚠️ OVERFITTING (Quá khớp)
```
*Lý do:* Việc sử dụng 50 đặc trưng trên một tập dữ liệu có lớp thiểu số ít khiến CatBoost học quá chi tiết.

#### **CẤP ĐỘ 3: Đi sâu (Phân tích lỗi FP/FN)**

**Ma trận nhầm lẫn trên tập Test:**
```
┌─────────────────────┐
│ TN:  992  FP:  380 │  (Lớp 0: No Match)
│ FN:  144  TP:  125 │  (Lớp 1: Match)
└─────────────────────┘
```

**Phân tích:**
- **FP (380):** Cao nhất - Gây nhiễu cho người dùng (khuyến nghị không hợp).
- **FN (144):** Bỏ lỡ Match - Chi phí cơ hội.
- **Chiến lược:** Mô hình hiện tại đang ưu tiên Recall hơn Precision để không bỏ lỡ Match.

#### **CẤP ĐỘ 4: Nguyên nhân gốc rễ (Kiểm tra chất lượng dữ liệu)**

**Phân tích ngoại lệ (Outlier Analysis):**
- Tỷ lệ ngoại lệ ở mẫu lỗi (0.04) THẤP HƠN mẫu đúng (0.10).
- **Kết luận:** Lỗi không phải do dữ liệu "bẩn", mà do các đặc trưng Pre-match chưa đủ mạnh để phân biệt các trường hợp khó.

#### **CẤP ĐỘ 5: Vận hành (Tính công bằng - Giới tính)**

| Phân đoạn | F1-score | Số mẫu |
|-----------|----------|--------|
| **Nữ (Female)** | **0.3571** | 833 |
| **Nam (Male)** | **0.2880** | 808 |

**Insight:** Mô hình hoạt động tốt hơn trên Nữ giới (~7%). Có thể hành vi/tiêu chuẩn của Nữ giới nhất quán hơn trong bộ dữ liệu này.

## 6. Kết quả & Kiểm chứng (Validation)

### ✅ Các kiểm tra xác thực đã vượt qua
- [x] Dashboard chẩn đoán lưu tại `plots/evaluation_diagnostics.png` ✓
- [x] Không phát hiện vấn đề chất lượng dữ liệu nghiêm trọng ở mẫu lỗi ✓
- [x] Tính công bằng giới tính ở mức chấp nhận được (chênh lệch < 10%) ✓
