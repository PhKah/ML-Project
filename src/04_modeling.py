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
# CONFIGURATION - STRICT VALIDATION (TEST SET HIDDEN)
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'test_data': 'Data/test_set_hidden.csv', # STRICT ISOLATION
        'results_csv': 'Data/modeling_results_val.csv',
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

if not os.path.exists(CONFIG['paths']['model_dir']):
    os.makedirs(CONFIG['paths']['model_dir'])

print("=" * 80)
print("TASK 04: MODEL SELECTION & TUNING (TEST SET HIDDEN)")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def prepare_and_hide_test(config):
    print(f"\n[1] Splitting data & Hiding Test Set (Strict Isolation)...")
    df = pd.read_csv(config['paths']['input_data'])
    
    X = df.drop(['iid', 'pid', 'match'], axis=1)
    y = df['match']
    
    # First split: Train vs Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config['experiment']['test_size'] + config['experiment']['val_size']), 
        random_state=config['experiment']['random_state'], stratify=y
    )
    
    # Second split: Val vs Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, 
        random_state=config['experiment']['random_state'], stratify=y_temp
    )
    
    # SAVE TEST SET TO DISK (HIDE IT FROM THIS SCRIPT)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(config['paths']['test_data'], index=False)
    
    print(f"   ✓ Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"   ✓ Test set (size={len(X_test)}) HIDDEN in {config['paths']['test_data']}")
    return X_train, X_val, y_train, y_val, X.columns.tolist()

def find_best_threshold(model, X_val, y_val):
    if not hasattr(model, "predict_proba"):
        return 0.5, f1_score(y_val, model.predict(X_val))
    
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_val, y_probs >= t) for t in thresholds]
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def run_tuning(X_train, X_val, y_train, y_val, config):
    print(f"\n[2] Performing Model Tuning on Validation set...")
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
        
        # Evaluate other metrics on Validation for selection
        val_probs = best_pipeline.predict_proba(X_val)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
        val_pred = (val_probs >= best_t) if val_probs is not None else best_pipeline.predict(X_val)
        
        results[name] = {
            'val_f1': val_f1,
            'val_prec': precision_score(y_val, val_pred, zero_division=0),
            'val_rec': recall_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_probs) if val_probs is not None else None,
            'best_threshold': best_t,
            'best_params': gs.best_params_,
            'model_pipeline': best_pipeline
        }
        print(f"      Val F1: {val_f1:.4f} (Threshold={best_t:.2f})")
        
    return results

def save_winner(results, config):
    # Select winner based on Validation F1
    winner_name = max(results, key=lambda x: results[x]['val_f1'])
    winner_data = results[winner_name]
    
    model_path = os.path.join(config['paths']['model_dir'], 'winner_model.joblib')
    joblib.dump({
        'name': winner_name,
        'pipeline': winner_data['model_pipeline'],
        'threshold': winner_data['best_threshold']
    }, model_path)
    
    print(f"\n✓ WINNER LOCKED: {winner_name}")
    print(f"✓ Model and Threshold saved to {model_path}")
    return winner_name

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, f_names = prepare_and_hide_test(CONFIG)
    results = run_tuning(X_train, X_val, y_train, y_val, CONFIG)
    
    winner = save_winner(results, CONFIG)
    
    # Save Validation Results Summary
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
    pd.DataFrame(summary).sort_values('Val_F1', ascending=False).to_csv(CONFIG['paths']['results_csv'], index=False)
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY (TEST SET REMAINS HIDDEN)")
    print("=" * 80)
    print(pd.DataFrame(summary).sort_values('Val_F1', ascending=False).to_string(index=False))
