# ❤ Speed Dating AI — Ứng dụng Đo Tỷ Lệ Tương Hợp

Web app học máy chạy **hoàn toàn cục bộ**, sử dụng toàn bộ pipeline ML từ dự án Speed Dating Experiment (Columbia, 2002–2004).

---

## Cấu trúc thư mục

```
speed_dating_app/
├── backend.py              ← Flask server + ML engine + API routes
├── requirements.txt
├── templates/
│   └── index.html          ← Frontend HTML/CSS/JS (963 dòng)
├── models/
│   └── winner_model.joblib ← CatBoost pipeline đã train (153 đặc trưng)
├── data/
│   ├── user_profiles.csv   ← 544 hồ sơ người dùng (presets)
│   ├── modeling_results_val.csv  ← So sánh 6 mô hình
│   └── learn_error.tsv     ← Đường cong học CatBoost
└── assets/
    ├── behavioral_gaps.png
    ├── evaluation.png
    └── counterfactual.png
```

## Cài đặt & Chạy

```bash
cd speed_dating_app

# (Khuyến nghị) tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python backend.py
```

Mở trình duyệt: **http://localhost:5000**

---

## 5 Tab chức năng

| Tab | Nội dung |
|-----|----------|
| ❤ **So Sánh** | Nhập 2 hồ sơ → gauge % + radar sở thích + phân tích yếu tố |
| 🔍 **Tìm Người Phù Hợp** | Batch-predict toàn bộ 544 hồ sơ → top N match |
| 📊 **Khám Phá Dữ Liệu** | KPI, biểu đồ hobby gaps, nghịch lý lựa chọn, insights |
| 🤖 **Kết Quả ML** | Bảng 6 mô hình, đường cong học, đồ thị từ dự án gốc |
| 📋 **Báo Cáo** | Full report nhóm: feature engineering, kết quả, insights |

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| `GET` | `/` | Web app chính |
| `POST` | `/api/predict` | Dự đoán compatibility 2 profile |
| `POST` | `/api/find-matches` | Tìm top N matches từ 544 hồ sơ |
| `GET` | `/api/stats` | Dataset statistics |
| `GET` | `/api/model-results` | Bảng so sánh 6 mô hình |
| `GET` | `/api/catboost-curve` | Đường cong học CatBoost |
| `GET` | `/api/hobby-gaps` | Hobby gap stats (match vs no-match) |
| `GET` | `/assets/<file>` | Serve plot images |
