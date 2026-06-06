import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# [ANTI-LEAKAGE] CONFIGURATION - PROFESSIONAL PIPELINE INTEGRATION
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'test_data': 'Data/test_set_hidden.csv',
        'results_csv': 'Data/modeling_results_val.csv',
        'model_dir': 'models/',
        'plot_dir': 'plots/'
    },
    'experiment': {
        'random_state': 42,
        'cv_splits': 5
    },
    'models': {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {'clf__C': [0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {'clf__max_depth': [5, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'params': {'clf__max_depth': [10, 20]}
        },
        'XGBoost': {
            'model': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'params': {
                'clf__max_depth': [3, 6],
                'clf__learning_rate': [0.05, 0.1]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(n_estimators=100, random_state=42, importance_type='gain', verbose=-1, class_weight='balanced'),
            'params': {
                'clf__num_leaves': [31, 63],
                'clf__learning_rate': [0.05, 0.1]
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0, auto_class_weights='Balanced'),
            'params': {
                'clf__depth': [4, 6],
                'clf__learning_rate': [0.05, 0.1]
            }
        }
    }
}

if not os.path.exists(CONFIG['paths']['model_dir']):
    os.makedirs(CONFIG['paths']['model_dir'])

print("=" * 80)
print("TASK 04: [LEAKAGE-FREE] PROFESSIONAL MODELING PIPELINE")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def prepare_data(config):
    print(f"\n[1] Loading Raw Enriched Data...")
    df = pd.read_csv(config['paths']['input_data'])
    
    if 'pair_id' not in df.columns:
        raise KeyError("Column 'pair_id' missing. Ensure src/03 was run correctly.")
        
    X = df.drop(['iid', 'pid', 'match'], axis=1) # Keep pair_id for split
    y = df['match']
    
    # Stratified Group Split for Test Set
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['experiment']['random_state'])
    train_val_idx, test_idx = next(sgkf.split(X, y, groups=df['pair_id']))
    
    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Save Test set (Hidden)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(config['paths']['test_data'], index=False)
    
    # Split train_val into train and val
    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config['experiment']['random_state'])
    train_idx, val_idx = next(sgkf_val.split(X_train_val, y_train_val, groups=X_train_val['pair_id']))
    
    X_train, y_train = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
    X_val, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]
    
    groups_train = X_train['pair_id']
    X_train = X_train.drop('pair_id', axis=1)
    X_val = X_val.drop('pair_id', axis=1)
    
    print(f"   ✓ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, y_train, y_val, groups_train

def build_pipeline(classifier, X):
    all_cols = set(X.columns.tolist())
    
    # 1. Định nghĩa nhóm Standard (Thặng dư & Biến liên tục không giới hạn)
    surplus_cols = {c for c in all_cols if '_surplus' in c}
    cont_cols = {'age', 'age_o', 'age_gap_calc'}
    standard_cols = (surplus_cols | cont_cols) & all_cols
    
    # 2. Định nghĩa nhóm MinMax (Khoảng cách & Thang điểm cố định 0-10)
    gap_cols = {c for c in all_cols if '_gap' in c} - standard_cols
    score_cols = {c for c in all_cols if c in [
        'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
        'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
        'movies', 'concerts', 'music', 'shopping', 'yoga',
        'exphappy', 'expnum', 'int_corr', 'imprace', 'imprelig'
    ] or (c.endswith('_o') and any(sub in c for sub in ['sports', 'dining', 'art', 'museums']))} - standard_cols
    minmax_cols = (gap_cols | score_cols) & all_cols
    
    # 3. Định nghĩa nhóm Categories (Nominal features cần One-Hot Encoding)
    # Bao gồm các biến định danh gốc và các biến đối tác (_o)
    cat_base = {'gender', 'race', 'goal', 'career_c', 'condtn', 'samerace'}
    cat_cols = {c for c in all_cols if any(b == c or f"{b}_o" == c for b in cat_base)} & all_cols
    
    # 4. Nhóm còn lại (Passthrough - giữ nguyên nhưng vẫn điền khuyết)
    remaining_cols = all_cols - standard_cols - minmax_cols - cat_cols
    
    # Xây dựng các Sub-pipelines
    minmax_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    standard_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    passthrough_pipe = SimpleImputer(strategy='median')

    # ColumnTransformer đảm bảo mỗi cột chỉ xuất hiện 1 lần
    transformer = ColumnTransformer([
        ('minmax', minmax_pipe, list(minmax_cols)),
        ('standard', standard_pipe, list(standard_cols)),
        ('cat', cat_pipe, list(cat_cols)),
        ('pass_imp', passthrough_pipe, list(remaining_cols))
    ])
    
    return Pipeline([
        ('preprocessor', transformer),
        ('smote', SMOTE(random_state=42)),
        ('clf', classifier)
    ])

def find_best_threshold(pipeline, X_val, y_val):
    if not hasattr(pipeline, "predict_proba"):
        return 0.5, fbeta_score(y_val, pipeline.predict(X_val), beta=0.5)
    
    y_probs = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [fbeta_score(y_val, y_probs >= t, beta=0.5, zero_division=0) for t in thresholds]
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

def run_tuning(X_train, X_val, y_train, y_val, groups_train, config):
    print(f"\n[2] Executing Leakage-Free GridSearchCV...")
    results = {}
    cv = StratifiedGroupKFold(n_splits=config['experiment']['cv_splits'], shuffle=True, random_state=42)
    
    for name, m_cfg in config['models'].items():
        print(f"    - Training {name}...")
        full_pipeline = build_pipeline(m_cfg['model'], X_train)
        gs = GridSearchCV(full_pipeline, m_cfg['params'], cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train, groups=groups_train)
        
        best_pipe = gs.best_estimator_
        best_t, val_f05 = find_best_threshold(best_pipe, X_val, y_val)
        
        val_probs = best_pipe.predict_proba(X_val)[:, 1] if hasattr(best_pipe, "predict_proba") else None
        val_pred = (val_probs >= best_t) if val_probs is not None else best_pipe.predict(X_val)
        
        results[name] = {
            'val_f05': val_f05,
            'val_f1': f1_score(y_val, val_pred),
            'val_prec': precision_score(y_val, val_pred, zero_division=0),
            'val_rec': recall_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_probs) if val_probs is not None else None,
            'best_threshold': best_t,
            'model_pipeline': best_pipe
        }
        print(f"      Val F0.5: {val_f05:.4f} (Threshold={best_t:.2f})")
        
    return results

def save_winner(results, config):
    winner_name = max(results, key=lambda x: results[x]['val_f05'])
    winner_data = results[winner_name]
    
    model_path = os.path.join(config['paths']['model_dir'], 'winner_model.joblib')
    joblib.dump({
        'name': winner_name,
        'pipeline': winner_data['model_pipeline'],
        'threshold': winner_data['best_threshold']
    }, model_path)
    
    print(f"\n✓ WINNER LOCKED: {winner_name}")
    print(f"✓ Pipeline saved to {model_path}")
    return winner_name

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, groups_train = prepare_data(CONFIG)
    results = run_tuning(X_train, X_val, y_train, y_val, groups_train, CONFIG)
    winner = save_winner(results, CONFIG)
    
    summary = []
    for name, res in results.items():
        summary.append({
            'Model': name,
            'Val_F05': res['val_f05'],
            'Val_F1': res['val_f1'],
            'Val_Prec': res['val_prec'],
            'Val_Rec': res['val_rec'],
            'Val_AUC': res['val_auc'],
            'Threshold': res['best_threshold']
        })
    pd.DataFrame(summary).sort_values('Val_F05', ascending=False).to_csv(CONFIG['paths']['results_csv'], index=False)
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY (LEAKAGE-FREE)")
    print("=" * 80)
    print(pd.DataFrame(summary).sort_values('Val_F05', ascending=False).to_string(index=False))
