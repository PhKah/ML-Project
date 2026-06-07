"""
backend.py — Speed Dating Compatibility App
Flask server + ML engine
Data source: data_final_v2.csv (từ repo ML-Project của nhóm)
Run: python backend.py  →  http://localhost:5000
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request, send_from_directory

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE, "models", "winner_model.joblib")
DATA_PATH    = os.path.join(BASE, "data",   "data_final_v2.csv")
RESULTS_PATH = os.path.join(BASE, "data",   "modeling_results_val.csv")
CURVE_PATH   = os.path.join(BASE, "data",   "learn_error.tsv")
ASSETS_DIR   = os.path.join(BASE, "assets")

# ─── Feature constants ─────────────────────────────────────────────────────
HOBBIES = ["sports","tvsports","exercise","dining","museums","art","hiking",
           "gaming","clubbing","reading","tv","theater","movies","concerts",
           "music","shopping","yoga"]
SURPLUS_ATTRS = ["attr","sinc","intel","fun","amb"]
ENTITY_COLS = [
    "age","gender","race","imprace","imprelig","goal","date","go_out",
    "career_c","exphappy","expnum",
    "attr1_1","sinc1_1","intel1_1","fun1_1","amb1_1","shar1_1",
    "attr2_1","sinc2_1","intel2_1","fun2_1","amb2_1","shar2_1",
    "attr3_1","sinc3_1","intel3_1","fun3_1","amb3_1",
    "attr4_1","sinc4_1","intel4_1","fun4_1","amb4_1","shar4_1",
    "attr5_1","sinc5_1","intel5_1","fun5_1","amb5_1",
] + HOBBIES

# ─── Label mappings ────────────────────────────────────────────────────────
GENDER_OPTS = {"0":"Nữ","1":"Nam"}
RACE_OPTS   = {"1":"Da đen","2":"Da trắng","3":"Latino","4":"Châu Á","5":"Bản địa Mỹ","6":"Khác"}
GOAL_OPTS   = {"1":"Một tối vui","2":"Gặp người mới","3":"Tìm hẹn hò",
               "4":"Quan hệ nghiêm túc","5":"Thử cho biết","6":"Khác"}
FREQ_OPTS   = {"1":"Vài lần/tuần","2":"2 lần/tuần","3":"1 lần/tuần",
               "4":"2 lần/tháng","5":"1 lần/tháng","6":"Vài lần/năm","7":"Gần như không"}
CAREER_OPTS = {"1":"Luật sư","2":"Học thuật/NC","3":"Tâm lý học","4":"Bác sĩ",
               "5":"Kỹ sư","6":"Nghệ thuật","7":"Tài chính/KD","8":"Bất động sản",
               "9":"Quốc tế/Nhân đạo","10":"Chưa quyết định","11":"Công tác XH",
               "12":"Ngôn ngữ trị liệu","13":"Chính trị","14":"Thể thao chuyên nghiệp",
               "15":"Khác","16":"Báo chí","17":"Kiến trúc"}
HOBBY_LABELS = {"sports":"Thể thao","tvsports":"Xem thể thao","exercise":"Tập gym",
                "dining":"Ăn ngoài","museums":"Bảo tàng","art":"Nghệ thuật",
                "hiking":"Leo núi","gaming":"Chơi game","clubbing":"Vũ trường",
                "reading":"Đọc sách","tv":"Xem TV","theater":"Kịch",
                "movies":"Phim ảnh","concerts":"Hòa nhạc","music":"Âm nhạc",
                "shopping":"Mua sắm","yoga":"Yoga"}
HOBBY_EMOJIS = {"sports":"⚽","tvsports":"📺","exercise":"💪","dining":"🍽",
                "museums":"🏛","art":"🎨","hiking":"🏕","gaming":"🎮",
                "clubbing":"💃","reading":"📚","tv":"📺","theater":"🎭",
                "movies":"🎬","concerts":"🎵","music":"🎸","shopping":"🛍","yoga":"🧘"}
ATTR_LABELS  = {"attr":"Ngoại hình","sinc":"Chân thành","intel":"Thông minh",
                "fun":"Vui tính","amb":"Tham vọng","shar":"Sở thích chung"}

# ─── Cache ─────────────────────────────────────────────────────────────────
_MODEL    = None
_DF       = None      # data_final_v2.csv đầy đủ
_PROFILES = None      # 1 dòng / người (groupby iid)

def _load_model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(MODEL_PATH)
        _MODEL = {"name": b["name"], "pipeline": b["pipeline"],
                  "threshold": float(b["threshold"]),
                  "feat_order": list(b["pipeline"].feature_names_in_)}
    return _MODEL

def _load_df() -> pd.DataFrame:
    """Load data_final_v2.csv (toàn bộ pairs)."""
    global _DF
    if _DF is None:
        _DF = pd.read_csv(DATA_PATH)
    return _DF

def _load_profiles() -> pd.DataFrame:
    """Trích xuất 1 hồ sơ / người từ data_final_v2.csv (groupby iid)."""
    global _PROFILES
    if _PROFILES is None:
        df = _load_df()
        # Lấy các cột entity có trong data
        avail = [c for c in ENTITY_COLS if c in df.columns]
        profiles = df.groupby("iid")[avail].first()
        profiles = profiles.fillna(profiles.median(numeric_only=True))
        _PROFILES = profiles
    return _PROFILES

# ─── Defaults & Samples ────────────────────────────────────────────────────
def get_default_profile() -> dict:
    prof = _load_profiles()
    med  = prof[ENTITY_COLS].median(numeric_only=True)
    out  = {}
    for c in ENTITY_COLS:
        v = float(med.get(c, 0))
        if c in ("gender","race","goal","date","go_out","career_c","age","expnum"):
            out[c] = int(round(v))
        else:
            out[c] = round(v, 2)
    return out

def get_sample_profiles() -> list[dict]:
    prof = _load_profiles()
    items = []
    for iid in list(prof.index):   # tất cả 544 người
        row = prof.loc[iid]
        g   = GENDER_OPTS.get(str(int(row.get("gender", 0))), "?")
        p   = {}
        for c in ENTITY_COLS:
            v = float(row.get(c, 0))
            if c in ("gender","race","goal","date","go_out","career_c","age","expnum"):
                p[c] = int(round(v))
            else:
                p[c] = round(v, 2)
        items.append({"label": f"Người #{int(iid)} · {g} · {int(row.get('age',25))} tuổi",
                      "profile": p})
    return items

# ─── Feature engineering ───────────────────────────────────────────────────
def build_features(a: dict, b: dict, condtn: int = 1) -> pd.DataFrame:
    row = {}
    for c in ENTITY_COLS:
        row[c]        = float(a.get(c, 0))
        row[f"{c}_o"] = float(b.get(c, 0))
    row["samerace"]      = int(int(a.get("race",0)) == int(b.get("race",0)))
    ai = np.array([float(a.get(h, 0)) for h in HOBBIES])
    bi = np.array([float(b.get(h, 0)) for h in HOBBIES])
    row["int_corr"]      = round(float(np.corrcoef(ai,bi)[0,1]),3) if ai.std()>0 and bi.std()>0 else 0.0
    row["condtn"]        = condtn
    row["age_gap_calc"]  = abs(float(a.get("age",25)) - float(b.get("age",25)))
    for h in HOBBIES:
        row[f"{h}_gap"]  = abs(float(a.get(h,0)) - float(b.get(h,0)))
    for at in SURPLUS_ATTRS:
        row[f"{at}_surplus_51_s"] = float(b.get(f"{at}5_1",5)) - (float(a.get(f"{at}1_1",50))/10.0)
        row[f"{at}_surplus_51_p"] = float(a.get(f"{at}5_1",5)) - (float(b.get(f"{at}1_1",50))/10.0)
        row[f"{at}_surplus_31_s"] = float(b.get(f"{at}3_1",5)) - (float(a.get(f"{at}1_1",50))/10.0)
        row[f"{at}_surplus_31_p"] = float(a.get(f"{at}3_1",5)) - (float(b.get(f"{at}1_1",50))/10.0)
    feat = _load_model()["feat_order"]
    return pd.DataFrame([row]).reindex(columns=feat)

# ─── Prediction ────────────────────────────────────────────────────────────
def predict_compatibility(a: dict, b: dict) -> dict:
    m    = _load_model()
    X    = build_features(a, b)
    prob = float(m["pipeline"].predict_proba(X)[0, 1])
    thr  = m["threshold"]
    pct  = round(prob * 100, 1)

    if   prob >= thr + 0.15: verdict = "Rất tương hợp ✨"
    elif prob >= thr:         verdict = "Có khả năng tương hợp 💫"
    elif prob >= thr - 0.10: verdict = "Tương hợp trung bình 🤔"
    else:                     verdict = "Ít tương hợp 💭"

    drivers = [
        {"label":"Cùng chủng tộc",
         "value":"Có" if a.get("race")==b.get("race") else "Không",
         "positive": a.get("race")==b.get("race")},
        {"label":"Chênh lệch tuổi",
         "value":f"{abs(float(a.get('age',25))-float(b.get('age',25))):.0f} tuổi",
         "positive": abs(float(a.get("age",25))-float(b.get("age",25))) <= 4},
        {"label":"Tương quan sở thích",
         "value":f"{float(X['int_corr'].iloc[0]):+.2f}",
         "positive": float(X["int_corr"].iloc[0]) >= 0.3},
        {"label":"Tổng khác biệt sở thích",
         "value":f"{sum(abs(float(a.get(h,5))-float(b.get(h,5))) for h in HOBBIES):.0f} điểm",
         "positive": sum(abs(float(a.get(h,5))-float(b.get(h,5))) for h in HOBBIES) <= 50},
    ]
    return {"percent":pct, "probability":prob, "threshold":round(thr*100,1),
            "is_match": prob>=thr, "verdict":verdict, "drivers":drivers,
            "hobby_a": [float(a.get(h,5)) for h in HOBBIES],
            "hobby_b": [float(b.get(h,5)) for h in HOBBIES],
            "hobby_labels": list(HOBBIES)}

# ─── Batch match finder ────────────────────────────────────────────────────
def find_best_matches(profile: dict, top_n: int = 10, opposite_gender: bool = True) -> list:
    prof = _load_profiles()
    if opposite_gender:
        tg   = 1 - int(profile.get("gender", 0))
        prof = prof[prof["gender"] == tg]

    m = _load_model()
    rows, iids = [], []
    for iid, row_b in prof.iterrows():
        b = {c: float(row_b.get(c, 0)) for c in ENTITY_COLS}
        r = {}
        for c in ENTITY_COLS:
            r[c]        = float(profile.get(c, 0))
            r[f"{c}_o"] = b[c]
        r["samerace"]     = int(int(profile.get("race",0)) == int(b.get("race",0)))
        ai = np.array([float(profile.get(h,0)) for h in HOBBIES])
        bi = np.array([b[h] for h in HOBBIES])
        r["int_corr"]     = float(np.corrcoef(ai,bi)[0,1]) if ai.std()>0 and bi.std()>0 else 0.0
        r["condtn"]       = 1
        r["age_gap_calc"] = abs(float(profile.get("age",25)) - b["age"])
        for h in HOBBIES:
            r[f"{h}_gap"] = abs(float(profile.get(h,5)) - b[h])
        for at in SURPLUS_ATTRS:
            r[f"{at}_surplus_51_s"] = b.get(f"{at}5_1",5) - (float(profile.get(f"{at}1_1",50))/10.0)
            r[f"{at}_surplus_51_p"] = float(profile.get(f"{at}5_1",5)) - (b.get(f"{at}1_1",50)/10.0)
            r[f"{at}_surplus_31_s"] = b.get(f"{at}3_1",5) - (float(profile.get(f"{at}1_1",50))/10.0)
            r[f"{at}_surplus_31_p"] = float(profile.get(f"{at}3_1",5)) - (b.get(f"{at}1_1",50)/10.0)
        rows.append(r)
        iids.append(iid)

    X_batch = pd.DataFrame(rows).reindex(columns=m["feat_order"])
    probs   = m["pipeline"].predict_proba(X_batch)[:, 1]

    results = []
    for iid, prob, row_b in zip(iids, probs, [prof.loc[i] for i in iids]):
        hobby_scores = {h: float(row_b.get(h, 5)) for h in HOBBIES}
        top3 = sorted(HOBBIES, key=lambda h: hobby_scores[h], reverse=True)[:3]
        results.append({
            "iid":      int(iid),
            "percent":  round(float(prob)*100, 1),
            "is_match": bool(float(prob) >= m["threshold"]),
            "gender":   int(row_b.get("gender", 0)),
            "age":      int(row_b.get("age", 25)),
            "career":   CAREER_OPTS.get(str(int(row_b.get("career_c",7))
                          if pd.notna(row_b.get("career_c")) else 7), "?"),
            "goal":     GOAL_OPTS.get(str(int(row_b.get("goal",1))
                          if pd.notna(row_b.get("goal")) else 1), "?"),
            "hobbies":  [HOBBY_EMOJIS[h] for h in top3],
        })
    results.sort(key=lambda x: x["percent"], reverse=True)
    return results[:top_n]

# ─── Data analysis — tính từ data_final_v2.csv thực ───────────────────────
def get_dataset_stats() -> dict:
    df   = _load_df()
    prof = _load_profiles()
    m    = _load_model()
    return {
        "total_pairs":    len(df),
        "total_users":    int(df["iid"].nunique()),
        "match_rate":     round(float(df["match"].mean()) * 100, 1),
        "total_matches":  int(df["match"].sum()),
        "male_count":     int((prof["gender"] == 1).sum()),
        "female_count":   int((prof["gender"] == 0).sum()),
        "avg_age_male":   round(float(prof[prof["gender"]==1]["age"].mean()), 1),
        "avg_age_female": round(float(prof[prof["gender"]==0]["age"].mean()), 1),
        "n_waves":        21,
        "model_name":     m["name"],
        "threshold":      round(m["threshold"] * 100, 1),
        "n_features":     len(m["feat_order"]),
        "val_auc":        0.644,
        "val_f05":        0.323,
    }

def get_hobby_gap_stats() -> dict:
    """Tính khoảng cách sở thích trung bình: matched vs unmatched — từ data thực."""
    df      = _load_df()
    matched = df[df["match"] == 1]
    notmatch= df[df["match"] == 0]
    gap_cols = [f"{h}_gap" for h in HOBBIES]
    avail    = [c for c in gap_cols if c in df.columns]
    avg_m    = [round(float(matched[c].mean()), 3)  if c in avail else 0.0 for c in gap_cols]
    avg_nm   = [round(float(notmatch[c].mean()), 3) if c in avail else 0.0 for c in gap_cols]
    return {"hobbies": HOBBIES,
            "labels":  [HOBBY_LABELS[h] for h in HOBBIES],
            "match":   avg_m,
            "nomatch": avg_nm}

def get_model_comparison() -> list[dict]:
    df = pd.read_csv(RESULTS_PATH)
    rows = []
    for _, r in df.iterrows():
        rows.append({"name": r["Model"],
                     "f05":  round(float(r["Val_F05"]),  4),
                     "f1":   round(float(r["Val_F1"]),   4),
                     "prec": round(float(r["Val_Prec"]), 4),
                     "rec":  round(float(r["Val_Rec"]),  4),
                     "auc":  round(float(r["Val_AUC"]),  4),
                     "thr":  round(float(r["Threshold"]),2)})
    return sorted(rows, key=lambda x: x["f05"], reverse=True)

def get_catboost_curve() -> dict:
    df = pd.read_csv(CURVE_PATH, sep="\t")
    return {"iterations": df["iter"].tolist(), "logloss": df["Logloss"].tolist()}

def get_field_groups() -> list[dict]:
    return [
        {"id":"demo","title":"Thông tin cơ bản","expanded":True,"fields":[
            {"key":"age",      "label":"Tuổi",                  "kind":"number","min":18,"max":55,"step":1},
            {"key":"gender",   "label":"Giới tính",             "kind":"select","options":GENDER_OPTS},
            {"key":"race",     "label":"Chủng tộc",             "kind":"select","options":RACE_OPTS},
            {"key":"career_c", "label":"Nghề nghiệp",           "kind":"select","options":CAREER_OPTS},
            {"key":"goal",     "label":"Mục tiêu",              "kind":"select","options":GOAL_OPTS},
            {"key":"date",     "label":"Tần suất hẹn",          "kind":"select","options":FREQ_OPTS},
            {"key":"go_out",   "label":"Tần suất ra ngoài",     "kind":"select","options":FREQ_OPTS},
            {"key":"imprace",  "label":"Cùng chủng tộc quan trọng","kind":"slider","min":0,"max":10,"step":1},
            {"key":"imprelig", "label":"Cùng tôn giáo quan trọng", "kind":"slider","min":0,"max":10,"step":1},
            {"key":"exphappy", "label":"Kỳ vọng hạnh phúc",    "kind":"slider","min":0,"max":10,"step":1},
        ]},
        {"id":"hobbies","title":"Sở thích (0–10)","expanded":True,"fields":[
            {"key":h,"label":HOBBY_LABELS[h],"kind":"slider","min":0,"max":10,"step":1}
            for h in HOBBIES
        ]},
        {"id":"pref","title":"Bạn tìm gì ở đối phương (~100 điểm)","expanded":False,"fields":[
            {"key":f"{a}1_1","label":ATTR_LABELS.get(a,a),"kind":"slider","min":0,"max":100,"step":1}
            for a in ["attr","sinc","intel","fun","amb","shar"]
        ]},
        {"id":"self","title":"Tự đánh giá bản thân (0–10)","expanded":False,"fields":[
            {"key":f"{a}3_1","label":ATTR_LABELS.get(a,a),"kind":"slider","min":0,"max":10,"step":1}
            for a in ["attr","sinc","intel","fun","amb"]
        ]},
    ]

# ─── Flask ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    app_data = {
        "schema":   get_field_groups(),
        "samples":  get_sample_profiles(),
        "defaults": get_default_profile(),
        "opts": {"gender":GENDER_OPTS,"race":RACE_OPTS,"goal":GOAL_OPTS,
                 "freq":FREQ_OPTS,"career":CAREER_OPTS,
                 "hobby_labels":HOBBY_LABELS,"hobby_emojis":HOBBY_EMOJIS},
        "model": {"name":      _load_model()["name"],
                  "threshold": round(_load_model()["threshold"]*100,1),
                  "n_features":len(_load_model()["feat_order"])},
    }
    return render_template("index.html", app_data=json.dumps(app_data, ensure_ascii=False))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        return jsonify({"ok": True, "result": predict_compatibility(data["profileA"], data["profileB"])})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/find-matches", methods=["POST"])
def api_find_matches():
    data = request.get_json()
    try:
        matches = find_best_matches(data["profile"],
                                    top_n=int(data.get("top_n", 10)),
                                    opposite_gender=bool(data.get("opposite_gender", True)))
        return jsonify({"ok": True, "matches": matches})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Speed Dating Compatibility App")
    print(f"  Data: data_final_v2.csv ({len(_load_df())} cặp, {_load_df()['iid'].nunique()} người)")
    print(f"  Mô hình: {_load_model()['name']} | threshold={_load_model()['threshold']:.2f}")
    print("  Mở trình duyệt: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
