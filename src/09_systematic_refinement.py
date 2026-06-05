import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - IDENTICAL TO TASK 04 FOR COMPARISON
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'output_refined_data': 'Data/data_refined.csv',
        'results_csv': 'Data/modeling_results_refined.csv',
        'model_dir': 'models/',
        'plot_dir': 'plots/'
    },
    'experiment': {
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        'cv_splits': 5
    },
    'models': {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'model__C': [0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'model__max_depth': [5, 10, 15]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'params': {'model__max_depth': [10, 20]}
        },
        'XGBoost': {
            'model': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'params': {
                'model__max_depth': [3, 6],
                'model__learning_rate': [0.05, 0.1]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(n_estimators=100, random_state=42, importance_type='gain', verbose=-1),
            'params': {
                'model__num_leaves': [31, 63],
                'model__learning_rate': [0.05, 0.1]
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
            'params': {
                'model__depth': [4, 6],
                'model__learning_rate': [0.05, 0.1]
            }
        }
    }
}

# ============================================================================
# FEATURE ENRICHMENT (The ONLY difference from Task 04)
# ============================================================================
def create_distilled_features(df):
    print("[1] Creating Distilled Features based on Counterfactual Insights...")
    df_new = df.copy()
    
    # Values based on src/08 tipping point analysis
    df_new['is_age_match'] = (df_new['age_gap_calc'] <= 2).astype(int)
    df_new['is_interest_match'] = (df_new['int_corr'] >= 0.25).astype(int)
    df_new['match_synergy'] = df_new['is_age_match'] * df_new['is_interest_match']
    
    print(f"   ✓ Added: is_age_match, is_interest_match, match_synergy")
    return df_new

# ============================================================================
# FUNCTIONS (IDENTICAL TO TASK 04)
# ============================================================================

def prepare_data(config):
    print(f"\n[2] Preparing data and splitting (Strict Comparison)...")
    raw_df = pd.read_csv(config['paths']['input_data'])
    df = create_distilled_features(raw_df)
    
    # Save refined data
    df.to_csv(config['paths']['output_refined_data'], index=False)
    
    X = df.drop(['iid', 'pid', 'match'], axis=1)
    y = df['match']
    
    # Identical splitting logic as Task 04
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config['experiment']['test_size'] + config['experiment']['val_size']), 
        random_state=config['experiment']['random_state'], stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, 
        random_state=config['experiment']['random_state'], stratify=y_temp
    )
    
    print(f"   ✓ Train: {len(X_train)} | Val: {len(X_val)} | Test (Hidden): {len(X_test)}")
    return X_train, X_val, y_train, y_val

def find_best_threshold(model, X_val, y_val):
    if not hasattr(model, "predict_proba"):
        return 0.5, f1_score(y_val, model.predict(X_val))
    
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_val, y_probs >= t) for t in thresholds]
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def run_tuning(X_train, X_val, y_train, y_val, config):
    print(f"\n[3] Performing GridSearchCV on Refined Data...")
    results = {}
    cv = StratifiedKFold(n_splits=config['experiment']['cv_splits'], shuffle=True, random_state=config['experiment']['random_state'])
    
    for name, m_cfg in config['models'].items():
        print(f"    - Tuning {name}...")
        
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=config['experiment']['random_state'])),
            ('model', m_cfg['model'])
        ])
        
        gs = GridSearchCV(pipeline, m_cfg['params'], cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        best_pipeline = gs.best_estimator_
        
        # Optimize on VALIDATION
        best_t, val_f1 = find_best_threshold(best_pipeline, X_val, y_val)
        
        # Evaluate other metrics
        val_probs = best_pipeline.predict_proba(X_val)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
        val_pred = (val_probs >= best_t) if val_probs is not None else best_pipeline.predict(X_val)
        
        results[name] = {
            'val_f1': val_f1,
            'val_prec': precision_score(y_val, val_pred, zero_division=0),
            'val_rec': recall_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_probs) if val_probs is not None else None,
            'best_threshold': best_t,
            'best_params': gs.best_params_
        }
        print(f"      Val F1: {val_f1:.4f} (Threshold={best_t:.2f})")
        
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("TASK 09: SYSTEMATIC REFINEMENT & COMPARISON")
    print("=" * 80)
    
    X_train, X_val, y_train, y_val = prepare_data(CONFIG)
    results = run_tuning(X_train, X_val, y_train, y_val, CONFIG)
    
    # Save Results Summary
    summary = []
    for name, res in results.items():
        summary.append({
            'Model': name,
            'Val_F1': res['val_f1'],
            'Val_Prec': res['val_prec'],
            'Val_Rec': res['val_rec'],
            'Val_AUC': res['val_auc'],
            'Threshold': res['best_threshold']
        })
    results_df = pd.DataFrame(summary).sort_values('Val_F1', ascending=False)
    results_df.to_csv(CONFIG['paths']['results_csv'], index=False)
    
    print("\n" + "=" * 80)
    print("REFINED BENCHMARK SUMMARY (Comparison Ready)")
    print("=" * 80)
    print(results_df.to_string(index=False))
