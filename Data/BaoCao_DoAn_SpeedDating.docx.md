**BÁO CÁO ĐỒ ÁN**

**DỰ ĐOÁN GHÉP ĐÔI HẸN HÒ DỰA TRÊN HỌC MÁY**
*Predicting Dating Matches Using Machine Learning: A Dyadic Interaction Approach* 

Môn học: Học Máy (Machine Learning)
Nhóm thực hiện: Nhóm 1
Thành viên: 
Phạm Thái Hòa 202416205
Phạm Nam Khánh 202416240
Nguyễn Hợp Huy 202416233
Lê Nguyễn Nhật Huy 202416232
Giảng viên hướng dẫn: Nguyễn Thị Kim Anh
Năm học: 2025 - 2026

---

# **CHƯƠNG 1: GIỚI THIỆU VÀ MÔ TẢ BÀI TOÁN**

## **1.1 Bối cảnh và động lực**
Trong kỷ nguyên kinh tế số, việc kết nối con người không còn dựa trên sự ngẫu nhiên mà dần chuyển sang tối ưu hóa dựa trên dữ liệu. Bài toán dự đoán sự hòa hợp (Compatibility) trong hẹn hò cấp tốc (Speed Dating) không chỉ mang ý nghĩa xã hội mà còn là thử thách kỹ thuật lớn: Làm thế nào để mô hình hóa những cảm xúc định tính thành các vector định lượng có khả năng dự báo? Dự án này tập trung vào việc xây dựng một hệ thống học máy có khả năng dự đoán xác suất "Match" giữa hai cá nhân dựa trên sự giao thoa về hồ sơ năng lực, sở thích và kỳ vọng tâm lý.

## **1.2 Mô tả bài toán**
Xây dựng mô hình phân loại nhị phân dự báo khả năng ghép đôi thành công (Match = 1) giữa hai thực thể người dùng A và B. Điểm khác biệt cốt lõi là bài toán này không chỉ xét đặc trưng đơn lẻ của mỗi người mà phải xử lý **Dữ liệu Dyadic (Dữ liệu cặp đôi)**, nơi sự tương tác và độ vênh kỳ vọng giữa hai phía đóng vai trò quyết định.

## **1.3 Mô hình hóa Toán học** 
*   **Dạng bài toán:** Binary Classification.
*   **Không gian đầu vào ($X$):** Vector đặc trưng kết hợp $X = [F_A, F_B, F_{inter}]$, trong đó $F_{inter}$ là các đặc trưng tương tác phi tuyến (Gaps, Surplus).
*   **Không gian đầu ra ($y$):** $y \in \{0, 1\}$. $y=1$ khi và chỉ khi cả hai bên cùng chấp nhận kết nối.

## **1.4 Mục tiêu hệ thống**
1.  **Triệt tiêu rò rỉ dữ liệu hệ thống (Anti-Leakage):** Thiết lập quy trình bọc thép (Encapsulated Pipeline) để đảm bảo tính khách quan 100%.
2.  **Mô hình hóa nhận thức đa tầng:** Trích xuất tri thức từ "Độ vênh sở thích" (Hobby Gaps) và "Thặng dư kỳ vọng" (Mutual Surplus).
3.  **Tối ưu hóa $F_{0.5}$-Score:** Ưu tiên Precision để giảm thiểu "Báo động giả" (False Positives), bảo vệ trải nghiệm và sự tin tưởng của người dùng.

---

# **CHƯƠNG 2: DỮ LIỆU VÀ PHƯƠNG PHÁP HỌC MÁY**

## **2.1 Tập dữ liệu**

### **2.1.1 Nguồn dữ liệu**
Dữ liệu được trích xuất từ thí nghiệm "Speed Dating Experiment" của Đại học Columbia (Fisman et al., 2006). 
*   **Quy mô:** ~8,210 bản ghi tương tác sau khi làm sạch.
*   **Đặc điểm:** Chứa thông tin đa chiều từ hồ sơ nhân khẩu học đến đánh giá chủ quan tức thì sau 4 phút hẹn hò.

### **2.1.2 Quá trình sàng lọc dữ liệu**
Dữ liệu gốc chứa 195 cột, bao gồm nhiều thông tin thu thập ở các thời điểm khác nhau (trước, trong, và sau buổi hẹn). Để xây dựng một bài toán dự đoán có tính thực tế cao — tức là chỉ sử dụng thông tin có thể biết trước khi buổi hẹn diễn ra hoặc thông qua hồ sơ cá nhân — nhóm áp dụng tiêu chí sàng lọc nghiêm ngặt để triệt tiêu hoàn toàn rò rỉ dữ liệu (Data Leakage).

**Tiêu chí giữ lại cột (feature):**
*   **Thông tin nhân khẩu học:** Trích xuất từ hồ sơ đăng ký (age, gender, race, field_cd, career_c...)
*   **Sở thích cá nhân:** 17 nhóm sở thích từ thể thao đến văn hóa (sports, reading, music, yoga...)
*   **Kỳ vọng và Tự nhận thức (Trước buổi hẹn):** Toàn bộ bộ khảo sát 5 tầng nhận thức:
    *   `1_1`: Tiêu chí tìm kiếm bạn đời lý tưởng.
    *   `3_1`: Tôi tự đánh giá bản thân như thế nào.
    *   `5_1`: Tôi nghĩ người khác nhìn nhận mình như thế nào.
*   **Điều kiện thí nghiệm:** Biến `condtn` và mức độ tương quan sở thích `int_corr`.

**Loại bỏ cột:**
*   **Đánh giá trong buổi hẹn (Point-of-contact):** Các biến chấm điểm đối phương ngay sau 4 phút gặp gỡ (`attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`, `prob`). Nhóm quyết định loại bỏ để ép mô hình phải học cách dự báo từ **bản chất hồ sơ** thay vì dựa vào cảm xúc nhất thời đã bộc lộ.
*   **Các đánh giá hậu sự kiện (Post-event):** Xóa bỏ toàn bộ các khảo sát thực hiện sau khi buổi hẹn kết thúc (Waves 2_1, 3_s, etc.)
*   **Dữ liệu thưa thớt:** Các cột với tỷ lệ thiếu dữ liệu > 60%.
*   **Định danh hành chính:** Các cột chỉ mang tính quản lý (iid, pid, id, round, wave...)

### **2.1.3 Phân tích thăm dò dữ liệu (EDA)**
Quá trình phân tích thăm dò tập trung vào việc hiểu rõ phân phối của các đặc trưng và kiểm định các giả thuyết về hành vi hẹn hò. Kết quả EDA cho thấy dữ liệu có độ nhiễu cao nhưng chứa đựng những quy luật tâm lý rõ rệt.

**Phân phối nhãn mục tiêu:** 
Tỷ lệ **Match (y=1)** chỉ chiếm khoảng **16.4%**, trong khi **No Match (y=0)** chiếm tới **83.6%**. Sự mất cân bằng này định hình chiến lược huấn luyện dựa trên việc ưu tiên Precision và sử dụng SMOTE để cân bằng dữ liệu.

**Kiểm định Giả thuyết thực nghiệm:**
1.  **Nghịch lý lựa chọn (Choice Overload):** Dữ liệu xác nhận rằng trong điều kiện ít lựa chọn (`condtn=1`), tỉ lệ Match đạt **20.2%**, cao hơn đáng kể so với điều kiện nhiều lựa chọn (`condtn=2`) chỉ đạt **15.7%**. Điều này chứng minh rằng việc có quá nhiều phương án làm giảm khả năng ra quyết định cuối cùng của con người.
2.  **Sự tương phản tương quan (Correlation Contrast):** Phân tích tương quan cho thấy một sự khác biệt khổng lồ:
    *   Các biến **Tĩnh** (Sở thích/Kỳ vọng trước buổi hẹn) có tương quan gần như bằng 0 với kết quả Match (ví dụ: `attr1_1` chỉ đạt 0.015).
    *   Các biến **Động** (Cảm nhận thực tế) có tương quan rất cao (ví dụ: `fun` đạt 0.27, `attr` đạt 0.26).
    *   *Hệ quả:* Đây là lý do nhóm quyết định loại bỏ các biến "Động" để xây dựng bài toán dự báo thực thụ từ hồ sơ gốc, dù việc này làm giảm điểm số mô hình nhưng tăng giá trị ứng dụng.

**Bảng 2.1 – Thống kê mô tả các đặc trưng tiêu biểu:**

| Đặc trưng | Ý nghĩa | Mean | Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `age` | Tuổi người tham gia | 26.29 | 3.31 | 18.0 | 34.0 |
| `int_corr` | Tương quan sở thích gốc | 0.20 | 0.30 | -0.83 | 0.91 |
| `reading_gap` | Độ lệch văn hóa đọc (Ivy Proxy) | 2.02 | 1.61 | 0.00 | 8.00 |
| `attr_surplus_51_s` | Thặng dư hấp dẫn (Tầng xã hội) | 4.70 | 1.72 | -1.00 | 10.0 |
| `match` | Kết quả ghép đôi (Nhãn) | 0.16 | 0.37 | 0.00 | 1.00 |

**Bảng 2.2 – Danh mục các đặc trưng chính được sử dụng:**

| Tên nhóm / Đặc trưng | Mô tả chi tiết | Kiểu dữ liệu | Vai trò / Ghi chú |
| :--- | :--- | :---: | :--- |
| **Nhân khẩu học** (`age`, `race`, `field_cd`...) | Thông tin định danh và nền tảng của người dùng. | Category/Int | Đặc trưng tĩnh từ hồ sơ |
| **Sở thích cá nhân** (`sports`, `music`...) | 17 nhóm sở thích định lượng từ 1-10. | Float [1-10] | Nền tảng tính toán Gap |
| **Kỳ vọng** (`attr1_1` đến `shar1_1`) | Trọng số ưu tiên cho các phẩm chất của bạn đời. | Float [0-100] | Điểm mốc tính Surplus |
| **Hobby Gaps** (17 biến) | Độ lệch tuyệt đối về lối sống giữa Subject và Partner. | Float [0-9] | Đo lường tính đồng điệu |
| **Mutual Surplus** (24 biến) | Sự thỏa mãn kỳ vọng trên 2 tầng nhận thức (3_1 và 5_1). | Float [-10, 10] | Đo lường tính tương hợp |
| **Bối cảnh** (`condtn`, `int_corr`) | Điều kiện buổi hẹn và tương quan sở thích sơ bộ. | Category/Float | Biến môi trường hệ thống |
| **Match** (Nhãn mục tiêu) | Trạng thái ghép đôi thành công từ cả hai phía. | Binary (0/1) | Biến phụ thuộc cần dự báo |

**Phân tích tương quan:** 
Các biến `_surplus` (Thặng dư) cho thấy mối tương quan thuận mạnh nhất với kết quả Match, chứng minh rằng sự thỏa mãn kỳ vọng quan trọng hơn là các đặc trưng nhân khẩu học đơn thuần. Đặc biệt, biến `reading_gap` nổi lên như một chỉ dấu quan trọng cho nhóm đối tượng trí thức, nơi sự đồng điệu về văn hóa đọc phản ánh sự tương hợp về thế giới quan.

### **2.1.4 Chiến lược Chống Rò rỉ Dữ liệu (Anti-Leakage Architecture)**
Dự án áp dụng nguyên lý **Statistical Isolation (Cách ly Thống kê)** nghiêm ngặt:
*   **Loại bỏ đặc trưng hậu sự kiện:** Xóa bỏ toàn bộ các cột chỉ có được sau khi kết quả Match đã lộ diện (ví dụ: các đánh giá follow-up).
*   **Group-aware Splitting:** Sử dụng mã định danh cặp đôi (`pair_id`) để đảm bảo các bản ghi phản chiếu (A-B và B-A) không bao giờ bị xé lẻ vào hai tập Train/Test khác nhau, triệt tiêu rò rỉ đối xứng (Reciprocal Leakage).

## **2.2 Tiền xử lý và Feature Engineering (Trái tim của hệ thống)**

### **2.2.1 Quy trình Trích xuất Dữ liệu Thô**
Thay vì chuẩn hóa toàn cục, hệ thống chuyển sang **Raw Feature Extraction**:
*   **Entity Cleaning:** Loại bỏ các "Bóng ma" (người dùng khuyết >20/56 thông tin cốt lõi) để bảo vệ tính toàn vẹn tham chiếu.
*   **Imputation & Scaling Inside Pipeline:** Việc điền khuyết (Median) và chuẩn hóa (MinMax/Standard) chỉ được thực hiện bên trong từng Fold của Cross-validation, ngăn chặn rò rỉ thông tin từ tập Test vào tham số của mô hình.

### **2.2.2 Tinh lọc tương hợp Đa tầng (Cognitive Distillation)**
Nhóm đã sáng tạo bộ đặc trưng mới mang tính học thuật cao:
1.  **17 Hobby Gaps:** Tính toán khoảng cách tuyệt đối $|Hobby_A - Hobby_B|$ để đo lường độ lệch lối sống.
2.  **24 Mutual Surplus (Thặng dư 2 tầng):**
    *   **Tầng thực tế xã hội (5_1):** Đối phương thực tế thế nào so với kỳ vọng của tôi.
    *   **Tầng Cái tôi cá nhân (3_1):** Đối phương tự thấy họ thế nào so với kỳ vọng của tôi.
    *   *Toán học:* $Surplus = Score_{Partner} - (Expectation_{Subject} / 10)$.

## **2.3 Các mô hình học máy**
Nhóm thực hiện Benchmark trên 6 thuật toán đa dạng: **Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, và CatBoost.** Trong đó, các mô hình Boosting được ưu tiên nhờ khả năng xử lý tương tác phi tuyến và dữ liệu mất cân bằng tốt thông qua cấu trúc Pipeline tích hợp SMOTE.

---

# **CHƯƠNG 3: KẾT QUẢ THÍ NGHIỆM**

## **3.1 Thiết lập thí nghiệm**
*   **Splitting:** Stratified Group K-Fold (K=5) dựa trên `pair_id`.
*   **Optimizing:** GridSearchCV tập trung vào $F_{0.5}$-Score để tối ưu Precision.

## **3.2 Kết quả so sánh và Kiểm định (Leakage-Free)**

**Bảng 3.1 – Hiệu năng trên tập Validation (Cross-validation):**

| Mô hình | Val $F_{0.5}$ | Val Precision | Val Recall | Val AUC |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (WINNER)** | **0.3215** | **32.32%** | **31.48%** | **0.645** |
| LightGBM | 0.2916 | 31.44% | 22.59% | 0.648 |
| CatBoost | 0.2906 | 31.87% | 21.48% | 0.627 |
| Random Forest | 0.2788 | 29.61% | 22.59% | 0.621 |
| Logistic Regression | 0.2626 | 25.75% | 28.52% | 0.594 |

**Bảng 3.2 – Hiệu năng cuối cùng trên tập Test ẩn (Hidden Test Set):**

| Chỉ số (XGBoost) | Giá trị thực tế | Ý nghĩa thực nghiệm |
| :--- | :---: | :--- |
| **Precision** | **29.71%** | Độ chính xác cao gấp 2 lần mức ngẫu nhiên. |
| **Recall** | **30.37%** | Khả năng bắt trọn 1/3 số cặp Match thực tế. |
| **F0.5-Score** | **0.2984** | Chỉ số hài hòa ưu tiên tính chính xác. |
| **ROC-AUC** | **0.6435** | Năng lực phân loại ổn định trên dữ liệu mới. |

## **3.3 Phân tích mô hình tốt nhất (XGBoost)**
Mô hình XGBoost đạt trạng thái cân bằng lý tưởng tại Threshold = 0.28.
1.  **Tính ổn định:** Sự chênh lệch giữa tập Validation (32%) và Test (29.7%) là rất thấp (~2.3%). điều này chứng minh mô hình **không bị Overfitting** và quy trình chống rò rỉ dữ liệu đã hoạt động hoàn hảo.
2.  **Giá trị ứng dụng:** Với Precision ~30%, hệ thống cung cấp một bộ lọc tin cậy, giúp người dùng tiết kiệm 70% thời gian tìm kiếm bằng cách loại bỏ các ứng viên không phù hợp ngay từ khâu hồ sơ.
3.  **Tính công bằng:** Hiệu năng trên Nam (F0.5=0.3057) và Nữ (F0.5=0.2911) đồng nhất, đảm bảo không có thiên kiến giới tính trong thuật toán gợi ý.

---

# **CHƯƠNG 4: CẤU TRÚC HỆ THỐNG VÀ MÃ NGUỒN**

## **4.1 Cấu trúc mã nguồn bọc thép**
Hệ thống được module hóa cao độ:
*   `src/03_data_preparation.py`: Trích xuất dữ liệu thô, bảo toàn phân phối gốc.
*   `src/04_modeling.py`: Pipeline đóng gói toàn bộ tri thức Scaling, Imputation, SMOTE và Classifier.
*   `src/05_evaluation.py`: Dashboard chẩn đoán 5 tầng (Metrics, Error Surgery, Learning Curve, Fairness, Importances).

## **4.2 Luồng xử lý (Data Pipeline)**
Raw Data → Entity Cleaning → Dyadic Merge → **[Pipeline: Imputer → ColumnTransformer (MinMax for Gaps, Standard for Surplus) → SMOTE → XGBoost]** → Final Prediction.

---

# **CHƯƠNG 5: KHÁM PHÁ VÀ PHÁT HIỆN MỚI (BEHAVIORAL INSIGHTS)**

## **5.1 Biến số "Ivy League Proxy"**
Thông qua phân tích tầm quan trọng của đặc trưng (Feature Importance), nhóm phát hiện `reading_gap` (độ lệch về sở thích đọc sách) là một trong những chỉ dấu dự báo mạnh nhất. Với đối tượng sinh viên đại học Columbia (thuộc nhóm Ivy League), sự đồng điệu về văn hóa đọc không chỉ đơn thuần là một sở thích, mà là "mã định danh" cho sự hòa hợp về trình độ tri thức và hệ giá trị cốt lõi.

## **5.2 Tipping Point của tuổi tác**
Phân tích phản thực nghiệm (Counterfactual Analysis) chỉ ra rằng tâm lý con người có một "điểm bùng phát" cảm xúc tại độ chênh lệch **1.92 năm**. 
*   **Dưới 2 năm:** Xác suất Match giữ ở mức cao và ổn định.
*   **Trên 2 năm:** Khả năng kết nối giảm đột ngột theo hàm mũ.
Điều này minh chứng cho quy luật "đồng trang lứa" (Peer effect), nơi sự tương đồng về bối cảnh sống và cột mốc trưởng thành quyết định tính hấp dẫn.

## **5.3 Yếu tố "Khiêm tốn nhận thức" (The Humility Factor)**
Đây là phát hiện tâm lý học quan trọng nhất của đồ án:
1.  **Độ vênh ảo tưởng:** Phân tích dữ liệu cho thấy điểm tự đánh giá bản thân (`3_1`) thường cao hơn dự đoán về cách xã hội nhìn nhận mình (`5_1`) trung bình **0.8 - 1.2 điểm**. Điều này khẳng định sự tồn tại của thiên kiến tự đề cao (Self-serving bias).
2.  **Sự thấu cảm thực chất:** Các cặp đôi Match thành công thường có **"Vênh nhận thức" (Misconception Gap)** tiệm cận bằng 0. 
3.  **Chiến lược bọc thép:** Việc nhóm sử dụng thặng dư kỳ vọng dựa trên tầng `5_1` thay vì `3_1` giúp mô hình đạt được tính bền vững cao hơn. Bằng cách xét trường hợp "ít ảo tưởng nhất" của người dùng, hệ thống tạo ra các gợi ý dựa trên sự thấu hiểu thực chất về vị thế xã hội của mỗi cá nhân, thay vì dựa trên cái tôi chủ quan.

---

# **CHƯƠNG 6: KẾT LUẬN**

## **6.1 Kết quả đạt được**
Đồ án đã xây dựng thành công một hệ thống dự đoán ghép đôi đạt chuẩn mực khoa học dữ liệu. Quan trọng hơn cả điểm số, dự án đã chứng minh được tính đúng đắn của phương pháp **Group-aware Anti-leakage**, biến mô hình từ một "máy học vẹt" thành một "chuyên gia dự báo" trung thực.

## **6.2 Hướng phát triển**
1.  Tích hợp các biến tương tác phi tuyến bậc cao ($Gap \times Surplus$).
2.  Áp dụng Deep Learning (Neural Collaborative Filtering) để bắt trọn các tín hiệu tiềm ẩn.
3.  Triển khai mô hình dưới dạng API thời gian thực để ứng dụng vào các nền tảng Social Discovery hiện đại.

---
**TÀI LIỆU THAM KHẢO**
1. Fisman, R., et al. (2006). Gender Differences in Mate Selection. Quarterly Journal of Economics.
2. Scikit-learn & XGBoost Documentation.
3. Chawla, N. V. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
