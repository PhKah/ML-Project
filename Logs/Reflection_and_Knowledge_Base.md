# TỔNG KẾT TƯ DUY VÀ TRI THỨC HỌC MÁY (PERSONAL REFLECTION)

File này lưu trữ những nguyên lý cốt lõi và tư duy giải quyết vấn đề mà tôi (người thực hiện) đã rút ra được qua dự án này. Đây là "tri thức nền" có thể áp dụng cho mọi bài toán học máy khác.

## 1. Tư duy về Dữ liệu (Data Mindset)
*   **Nguyên lý "Static vs Dynamic":** Trong thực tế, có những thứ ta biết trước (Sở thích) và những thứ chỉ nảy sinh trong quá trình tương tác (Đánh giá). Một mô hình học máy tốt phải phân biệt được hai loại này để tránh **Data Leakage**.
*   **Bản chất Quan hệ (Dyadic Data):** Biến `match` không thuộc về riêng ai, nó thuộc về một **cặp bài trùng**. Do đó, mô hình không nên chỉ nhìn vào dữ liệu đơn lẻ mà phải tập trung vào các **Đặc trưng tương đối** (như độ lệch tuổi, độ tương đồng sở thích). Việc chuyển đổi từ dữ liệu hai người thành một chỉ số "tương hợp" duy nhất là cách hiệu quả nhất để giảm độ phức tạp nhưng tăng sức mạnh dự báo.
*   **Feature Engineering - Từ "Liệt kê" sang "Tương quan":** Thay vì đưa hàng chục biến sở thích cá nhân (Hobbies như Sports, Art, Music...) vào mô hình làm bùng nổ số chiều, tư duy tinh tế hơn là tính toán **Độ tương đồng sở thích** giữa hai người. Một biến số duy nhất đại diện cho sự "đồng điệu" thường có giá trị dự báo cao hơn nhiều so với từng sở thích riêng lẻ.

*   **Tư duy Thực thể - Quan hệ (Entity-Relationship Cleaning):** Làm sạch dữ liệu không phải là một quy trình cồng kềnh cho toàn bộ bảng. Tri thức cao cấp là tách biệt xử lý:
    *   **Dữ liệu Thực thể (User):** Ưu tiên điền khuyết (Imputation) để bảo toàn hồ sơ người dùng.
    *   **Dữ liệu Quan hệ (Interaction):** Ưu tiên tính xác thực; nếu thiếu hụt quá nhiều thông tin về sự tương tác, việc xóa bỏ bản ghi sẽ tốt hơn là điền vào những con số "trung bình" vô hồn, giúp mô hình tập trung vào những tương tác có thật và chất lượng.

## 2. Tư duy về Đánh giá và Lựa chọn Mô hình (Evaluation & Modeling Mindset)
*   **Cái bẫy của Accuracy và Nghịch lý độ chính xác (Accuracy Paradox):** Trong một thế giới mất cân bằng, Accuracy cao thường là một "lời nói dối". Nếu lớp đa số chiếm 84%, một mô hình đoán bừa cũng có thể đạt 84% Accuracy. 
*   **F1-Score - Tiếng nói của sự thật:** Đây là "la bàn" giúp ta nhìn thẳng vào hiệu năng của lớp thiểu số (match=1). Việc F1-score thấp (0.31) thường kéo theo sự sụt giảm của cả **Recall** (khả năng bỏ lỡ cơ hội) và **Precision** (khả năng dự báo sai). Điều này phản ánh độ khó của bài toán: Các thuộc tính số học chỉ mới chạm được vào "phần nổi" của các quyết định cảm tính con người.
*   **Chiến lược "Cân bằng khi học, Khách quan khi thi":** Với dữ liệu lệch nhãn, việc chia tỷ lệ 6:2:2 máy móc là chưa đủ. Tri thức thực tế là phải can thiệp vào tập Huấn luyện (bằng Oversampling/Undersampling) để ép mô hình "học" lớp hiếm, nhưng giữ nguyên bản chất "lệch" ở tập Kiểm thử để đánh giá đúng thực tế.
*   **Sức mạnh của Gradient Boosting (XGBoost/CatBoost/LightGBM):** Mặc dù Decision Tree mang lại khả năng giải thích tốt (Interpretability), nhưng trong thực chiến dữ liệu dạng bảng (Tabular Data), họ hàng nhà Gradient Boosting mới là "vua". Các thuật toán Boosting có khả năng tối ưu hóa hàm suy hao qua từng bước, giúp xử lý mất cân bằng lớp cực tốt và mang lại bảng xếp hạng **Feature Importance đáng tin cậy và sắc nét nhất**, vượt trội so với các cây độc lập. Đưa các mô hình này vào so sánh tạo ra một khung Benchmark hoàn hảo từ cơ bản đến nâng cao.

## 3. Tư duy về Quy trình (Workflow Mindset)
*   **Tiêu chuẩn hóa (Standardization):** Học máy rất nhạy cảm với thang đo. Luôn đặt câu hỏi: "Các thuộc tính của mình đã đứng trên cùng một mặt phẳng chưa?" trước khi đưa vào mô hình.
*   **Chia để trị (Task-Oriented):** Việc đánh số code (`01_`, `02_`,...) và ghi log không phải là thủ tục hành chính, mà là cách để bảo vệ tư duy của mình khỏi sự lộn xộn khi bài toán trở nên phức tạp.

## 4. Những "Aha Moments" từ dự án
*   **Nghịch lý lựa chọn:** Khi có quá nhiều sự lựa chọn, bộ não con người có xu hướng trở nên khắt khe hơn. Điều này đã được chứng minh thực nghiệm qua sự sụt giảm tỷ lệ match trong dữ liệu.
*   **Biến ẩn `age_diff`:** Đôi khi những biến ta tự tạo ra (Feature Engineering) lại có sức mạnh hơn cả những biến có sẵn. Sự tương đồng về tuổi tác là một lực hút thầm lặng nhưng cực kỳ mạnh mẽ.

## 5. Tư duy về Tiền xử lý dữ liệu (Data Preprocessing Wisdom)

### 5.1 **Scaling Strategy - "Không phải mọi thứ đều cần một cách"**
*   **Bản chất:** Các cột khác nhau có bản chất khác nhau → cần scaling khác nhau.
*   **Thực hiện:**
    *   **MinMax Scaling [0,1]:** Dành cho dữ liệu có thang đo rõ ràng (1-10 như hobbies, ratings). Bảo toàn ý nghĩa "phần trăm" (0=không thích, 1=yêu thích).
    *   **StandardScaler (Z-score):** Dành cho dữ liệu liên tục (tuổi, thu nhập). Trung tâm hóa quanh mean=0, std=1, hiệu quả cho các thuật toán nhạy cảm với khoảng cách (SVM, KNN, Linear models).
*   **Nguyên lý:** Không có một cách scaling phổ quát. Hiểu bản chất dữ liệu → chọn phương pháp phù hợp.

### 5.2 **Dimensionality Reduction via Meaningful Aggregation**
*   **Bản chất:** Thay vì loại bỏ hoặc giảm chiều máy móc, hãy **gộp có chủ ý** dựa trên ngữ cảnh thực tế.
*   **Thực hiện:** Gộp 17 hobbies → 5 feature dựa trên **correlation heatmap**:
    *   `fitness_sport`: sports + tvsports + exercise (cùng "thể chất")
    *   `fine_arts`: museums + art + reading (cùng "trí tuệ/văn hóa")
    *   `entertainment`: theater + movies + music + concerts + tv (cùng "giải trí bị động")
    *   `social_nightlife`: dining + shopping + clubbing + gaming (cùng "xã hội/vui chơi")
    *   `outdoor_wellness`: hiking + yoga (cùng "bề ngoài/sức khỏe")
*   **Lợi ích:** 
    *   Giảm chiều từ 17→5 (70% giảm noise)
    *   Tăng khả năng giải thích (5 biến có ý nghĩa thực tế hơn 17 biến)
    *   Dễ calibration, tăng robustness
*   **Nguyên lý:** Dimensionality reduction tốt nhất không phải PCA máy móc, mà là "**aggregation có meaning**".

### 5.3 **Order-Dependent Feature Engineering - "Tuần tự vấn đề"**
*   **Bản chất:** Một số biến phụ thuộc vào trạng thái của biến khác. Phải xác định rõ thứ tự để tránh lỗi logic.
*   **Ví dụ cụ thể:**
    *   **A2 (age_map) PHẢI trước A7 (scaling):** Nếu scale age trước, sau đó tính age_gap sẽ tính từ age đã chuẩn hóa (Z-score) → kết quả sai lệch.
    *   **B1 đồng bộ user PHẢI sau A1 xóa user:** Không thể xóa interaction của 9 user nếu chưa biết ai bị xóa.
*   **Nguyên lý:** Coi feature engineering là một **đồ thị phụ thuộc (DAG)**, không phải một danh sách tuyến tính ngây thơ.

### 5.4 **Precision vs Pragmatism in Missing Data - "Entity ≠ Relationship"**
*   **Bản chất:** Chiến lược xử lý missing không nên một câu lệnh cho toàn bộ.
*   **Phân biệt rõ:**
    *   **Entity data (Users):** Ưu tiên **IMPUTATION** (điền trung bình/median)
        *   Lý do: Một người dùng là một thực thể có giá trị nội tại. Xóa họ = mất thông tin về người đó.
        *   Chấp nhận được tính "trung bình" vì dữ liệu user là "tĩnh" (không phụ thuộc quan hệ).
    *   **Relationship data (Interactions):** Ưu tiên **DELETION** (xóa hàng nếu quá thiếu)
        *   Lý do: Một mối quan hệ/tương tác phải "chân thực". Điền trung bình = "tương tác giả tạo" → gây nhiễu.
        *   Ngưỡng: Nếu một block rating (7 cột) thiếu > 50%, xóa hàng vì quá thiếu thông tin xác thực.
*   **Nguyên lý:** Không phải "one-size-fits-all". Hiểu **ngữ nghĩa** của dữ liệu (entity vs relationship) → chọn chiến lược phù hợp.

### 5.5 **Outlier Handling Philosophy - "Clip > Drop"**
*   **Bản chất:** Xóa outlier cứng nhắc sẽ làm mất dữ liệu. Clip (giới hạn) bảo toàn kích thước dataset.
*   **Thực hiện:** IQR clip thay vì drop
    *   Tính Q1, Q3, IQR
    *   Xác định cận: `lo = Q1 - 1.5×IQR`, `hi = Q3 + 1.5×IQR`
    *   Clip giá trị ngoài cận chứ không xóa hàng
*   **Lợi ích:** 
    *   Vẫn giữ thông tin "bị ngoại lai" nhưng không để nó méo mó mô hình
    *   Không mất bất kỳ hàng nào → dataset vẫn nguyên vẹn
*   **Nguyên lý:** Outlier không phải là "lỗi" cần xóa, mà là "nhiễu" cần kiềm chế.

### 5.6 **Data Synchronization in Dyadic Models - "Xóa phải đồng bộ"**
*   **Bản chất:** Khi dữ liệu có quan hệ nhị phân (cặp), xóa một thực thể PHẢI đồng bộ hóa quan hệ.
*   **Ví dụ:** Nếu xóa 9 user vì quá nhiều missing, PHẢI xóa tất cả interaction của 9 người đó.
    *   Nếu không: model sẽ thấy interaction với người "không tồn tại" → data corruption
    *   Thực hiện: `dropped_iids = {user_ids_deleted}`; sau đó `df_inter[df_inter['iid'].isin(dropped_iids)]`
*   **Nguyên lý:** Entity-Relationship model yêu cầu **tính toàn vẹn tham chiếu (referential integrity)**.

---
**Tri thức này là của tôi. Tôi đã trải nghiệm nó qua từng dòng code và biểu đồ.**
