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

**Kiểm toán tính toàn vẹn (Entity Integrity Audit):**
Trước khi trích xuất đặc trưng, nhóm thực hiện kiểm tra chất lượng hồ sơ ở cấp độ thực thể. Kết quả phát hiện **7 "bóng ma" (Ghost users)** khuyết thiếu >80% thông tin cốt lõi. Nhóm quyết định loại bỏ toàn bộ **79 bản ghi tương tác** liên quan đến các đối tượng này để bảo vệ tính xác thực của phân phối dữ liệu, đưa quy mô tập dữ liệu về con số ~8,210 bản ghi tin cậy.

**Tiêu chí giữ lại cột (feature):**
*   **Thông tin nhân khẩu học:** Trích xuất từ hồ sơ đăng ký (age, gender, race, field_cd, career_c...)
*   **Sở thích cá nhân:** 17 nhóm sở thích định lượng từ 1-10 (sports, reading, music, yoga...)
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
Quá trình phân tích thăm dò tập trung vào việc hiểu rõ bản chất của các đặc trưng gốc và kiểm định các quy luật hành vi. Đây là giai đoạn quan trọng để nhóm xác định các "vùng trũng thông tin" và đề xuất giải pháp xử lý dữ liệu.

**Phân phối nhãn mục tiêu:** 
Tỷ lệ **Match (y=1)** chỉ chiếm khoảng **16.5%**, trong khi **No Match (y=0)** chiếm tới **83.5%**. Sự mất cân bằng này cho thấy việc dự đoán ghép đôi thành công là bài toán khó, đòi hỏi mô hình phải có khả năng phân tách cực tốt.

**Kiểm định Giả thuyết thực nghiệm:**
1.  **Nghịch lý lựa chọn (Choice Overload):** Dữ liệu xác nhận rằng trong điều kiện ít lựa chọn (`condtn=1`), tỉ lệ Match đạt **20.2%**, cao hơn đáng kể so với điều kiện nhiều lựa chọn (`condtn=2`) chỉ đạt **15.7%**. 
2.  **Sự tương phản tương quan (Correlation Contrast):** Phân tích cho thấy các biến tĩnh (hồ sơ) có tương quan với nhãn rất thấp ($r \approx 0.026$), trong khi biến động (cảm nhận sau gặp mặt) tương quan rất cao (0.26 - 0.28). Đây là bằng chứng thực nghiệm để nhóm loại bỏ các biến động gây rò rỉ dữ liệu.

**Bảng 2.1 – Danh mục các Đặc trưng Gốc (Raw Feature Dictionary):**

| Nhóm đặc trưng | Tên biến tiêu biểu | Ý nghĩa thực tế | Vai trò trong mô hình |
| :--- | :--- | :--- | :--- |
| **Nhân khẩu học** | `age`, `race`, `field_cd` | Lý lịch và bối cảnh cá nhân. | Đặc trưng tĩnh nền tảng |
| **Lối sống** | `go_out`, `goal`, `date` | Thói quen và mục đích xã hội. | Biến bối cảnh hành vi |
| **Sở thích** | `sports`, `music`, `art` | 17 nhóm sở thích định lượng (1-10). | Cơ sở tính toán độ đồng điệu |
| **Kỳ vọng (1_1)** | `attr1_1` đến `shar1_1` | Tiêu chí lựa chọn bạn đời lý tưởng. | Điểm mốc tính toán thặng dư |
| **Cái tôi (3_1)** | `attr3_1` đến `amb3_1` | Cách cá nhân tự nhìn nhận bản thân. | Phân tích độ vênh nhận thức |
| **Dự đoán (5_1)** | `attr5_1` đến `amb5_1` | Cách cá nhân nghĩ xã hội nhìn mình. | Đặc trưng khiêm tốn xã hội |
| **Môi trường** | `condtn`, `int_corr` | Điều kiện buổi hẹn và tương quan gốc. | Biến hệ thống thí nghiệm |

**Định hướng Feature Engineering:** 
Dựa trên kết quả EDA cho thấy các biến đơn lẻ có tương quan rất yếu, nhóm đề xuất chiến lược: **Chuyển đổi từ dữ liệu Thực thể sang dữ liệu Tương tác** ở giai đoạn tiếp theo. Cụ thể, nhóm sẽ sử dụng 17 biến Sở thích để tạo ra các **Hobby Gaps** và dùng 3 tầng nhận thức (1_1, 3_1, 5_1) để tạo ra các **Mutual Surplus**.

**Bảng 2.3 – Kết quả kiểm định giả thuyết thống kê (H1, H3, H4):**

| Giả thuyết | Loại kiểm định | $p-value$ | Kết luận học thuật |
| :--- | :--- | :---: | :--- |
| **H1 (Nghịch lý lựa chọn)** | **Chi-square** | $3.08 \times 10^{-5}$ | Số lượng lựa chọn ảnh hưởng rõ rệt đến tỉ lệ Match ($p < 0.05$). |
| **H3 (Độ lệch tuổi tác)** | **Independent T-test** | $3.98 \times 10^{-10}$ | Các cặp Match có xu hướng gần tuổi nhau hơn một cách có hệ thống. |
| **H4 (Yếu tố Khiêm tốn)** | **Paired T-test** | $1.68 \times 10^{-24}$ | Sự khác biệt giữa tự đánh giá và dự báo xã hội là thực chất ($p \approx 0$). |

#### **🛡️ Kết quả Kiểm toán Rò rỉ Dữ liệu (Giả thuyết H2)**
Để đảm bảo mô hình có tính ứng dụng thực tế cao, nhóm đã thực hiện so sánh hệ số tương quan trung bình ($r$) của hai nhóm biến đối với nhãn mục tiêu:

| Nhóm đặc trưng | Tương quan trung bình ($r$) | Khả năng thu thập thực tế |
| :--- | :---: | :--- |
| **Biến Tĩnh** (Hồ sơ/Kỳ vọng) | **0.026** | Có sẵn ngay khi người dùng đăng ký. |
| **Biến Động** (Cảm nhận sau date) | **0.214** | Chỉ có sau khi buổi hẹn kết thúc. |

**Kết luận từ H2:** Biến động có tương quan mạnh gấp **8.2 lần** biến tĩnh. Tuy nhiên, việc sử dụng chúng sẽ gây **Rò rỉ dữ liệu (Data Leakage)** trầm trọng. Nhóm quyết định loại bỏ 100% biến động để xây dựng một hệ thống **Dự báo thực thụ** có khả năng hoạt động ngay khi người dùng chưa gặp mặt.

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

---

# **CHƯƠNG 6: THỬ THÁCH VÀ PHƯƠNG PHÁP GIẢI QUYẾT**

Trong quá trình thực hiện đồ án, nhóm đã đối mặt với nhiều thách thức mang tính đặc thù của dữ liệu xã hội và học máy thực tiễn. Dưới đây là các vấn đề trọng yếu và cách nhóm đã vượt qua:

#### **6.1 Thách thức về Rò rỉ Dữ liệu (Data Leakage)**
*   **Vấn đề:** Ban đầu, các bước chuẩn hóa (Scaling) và điền khuyết (Imputation) được thực hiện trên toàn bộ dataset trước khi chia tập Test. Điều này khiến mô hình "biết trước" các thông số thống kê của tập Test, dẫn đến điểm số cao ảo (over-optimistic).
*   **Giải pháp:** Nhóm đã đại tu kiến trúc sang dạng **Pipeline Encapsulation**. Toàn bộ tri thức về chuẩn hóa chỉ được học (`fit`) trên tập Train và sau đó mới áp dụng (`transform`) lên tập Test thông qua `imblearn.pipeline`.

#### **6.2 Rò rỉ Đối xứng (Reciprocal Leakage)**
*   **Vấn đề:** Trong dữ liệu cặp đôi, bản ghi A-B và B-A chứa thông tin phản chiếu nhau. Nếu chia tách ngẫu nhiên, mô hình sẽ "học thuộc lòng" kết quả từ tập Train để nhắc bài cho tập Test.
*   **Giải pháp:** Áp dụng kỹ thuật **Group-aware Splitting**. Nhóm tạo ra mã định danh cặp đôi duy nhất (`pair_id`) và sử dụng thuật toán `StratifiedGroupKFold` để đảm bảo một cặp đôi luôn nằm trong cùng một tập dữ liệu.

#### **6.3 Nghịch lý toán học trong Scaling**
*   **Vấn đề:** Các biến "Gap" (Khoảng cách) bản chất là trị tuyệt đối ($\ge 0$). Việc dùng `StandardScaler` sẽ tạo ra các giá trị âm, phá vỡ ý nghĩa vật lý của đặc trưng.
*   **Giải pháp:** Sử dụng **Hybrid Scaling (Đa bộ biến đổi)**. Nhóm dùng `ColumnTransformer` để áp dụng `MinMaxScaler` riêng cho nhóm Gaps (giữ tính không âm) và `StandardScaler` cho nhóm Surplus (mô tả sự lệch pha âm/dương).

#### **6.4 Vấn đề "Người dùng bóng ma" (Ghost Users)**
*   **Vấn đề:** Một số đối tượng tham gia khuyết thiếu tới 80% thông tin hồ sơ. Việc điền khuyết bằng trung vị cho các ca này sẽ tạo ra dữ liệu nhiễu cực lớn.
*   **Giải pháp:** Thực hiện **Entity Integrity Audit**. Nhóm kiên quyết loại bỏ 7 "bóng ma" và 79 tương tác liên quan để bảo vệ độ sạch của phân phối dữ liệu gốc.

---

# **CHƯƠNG 7: KẾT LUẬN**

## **7.1 Kết quả đạt được**
Đồ án đã xây dựng thành công một hệ thống dự đoán ghép đôi đạt chuẩn mực khoa học dữ liệu. Quan trọng hơn cả điểm số, dự án đã chứng minh được tính đúng đắn của phương pháp **Group-aware Anti-leakage**, biến mô hình từ một "máy học vẹt" thành một "chuyên gia dự báo" trung thực.

## **7.2 Hướng phát triển**
1.  Tích hợp các biến tương tác phi tuyến bậc cao ($Gap \times Surplus$).
2.  Áp dụng Deep Learning (Neural Collaborative Filtering) để bắt trọn các tín hiệu tiềm ẩn.
3.  Triển khai mô hình dưới dạng API thời gian thực để ứng dụng vào các nền tảng Social Discovery hiện đại.

---
**TÀI LIỆU THAM KHẢO**
1. Fisman, R., et al. (2006). Gender Differences in Mate Selection. Quarterly Journal of Economics.
2. Scikit-learn & XGBoost Documentation.
3. Chawla, N. V. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
