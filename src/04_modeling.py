import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, 
                             confusion_matrix, roc_auc_score, classification_report)

# Imbalanced handling
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Advanced Boosting Models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'results_csv': 'Data/modeling_results.csv',
        'plot_dir': 'plots/'
    },
    'experiment': {
        'test_size': 0.2,
        'val_size': 0.2, # 20% Val, 20% Test, 60% Train
        'random_state': 42,
        'cv_splits': 5
    },
    'models': {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {'C': [0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {'max_depth': [5, 10, 15]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'params': {'max_depth': [10, 20]}
        }
    }
}

# Add Advanced Boosting if available
if HAS_XGB:
    CONFIG['models']['XGBoost'] = {
        'model': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
        'params': {
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [5]
        }
    }

if HAS_LGB:
    CONFIG['models']['LightGBM'] = {
        'model': LGBMClassifier(n_estimators=100, random_state=42, importance_type='gain', verbose=-1),
        'params': {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'is_unbalance': [True]
        }
    }

if HAS_CAT:
    CONFIG['models']['CatBoost'] = {
        'model': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0, auto_class_weights='Balanced'),
        'params': {
            'depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
    }

print("=" * 80)
print("TASK 04: EXTENDED MODELING BENCHMARK")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def prepare_data(config):
    print(f"\n[1] Loading and Splitting data (Honest sequence)...")
    df = pd.read_csv(config['paths']['input_data'])
    
    X = df.drop(['iid', 'pid', 'match'], axis=1)
    y = df['match']
    feature_names = X.columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config['experiment']['test_size'] + config['experiment']['val_size']), 
        random_state=config['experiment']['random_state'], stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, 
        random_state=config['experiment']['random_state'], stratify=y_temp
    )

    print(f"   - Train set: {len(X_train)} rows ({y_train.mean()*100:.1f}% Match)")
    print(f"   - Val set:   {len(X_val)} rows ({y_val.mean()*100:.1f}% Match)")
    print(f"   - Test set:  {len(X_test)} rows ({y_test.mean()*100:.1f}% Match)")

    if HAS_SMOTE:
        print(f"\n[2] Applying SMOTE to TRAIN set ONLY...")
        sm = SMOTE(random_state=config['experiment']['random_state'])
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"   - Resampled Train: {len(X_train_res)} rows ({y_train_res.mean()*100:.1f}% Match)")
        return X_train_res, X_val, X_test, y_train_res, y_val, y_test, feature_names
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def find_best_threshold(model, X_val, y_val):
    if not hasattr(model, "predict_proba"):
        return 0.5, 0.0
    
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_scores = [f1_score(y_val, y_probs >= t) for t in thresholds]
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, config):
    print(f"\n[3] Training {len(config['models'])} models (Optimizing Class 1 F1)...")
    results = {}
    
    cv = StratifiedKFold(n_splits=config['experiment']['cv_splits'], shuffle=True, random_state=config['experiment']['random_state'])
    
    for name, m_cfg in config['models'].items():
        print(f"    - Tuning {name}...")
        gs = GridSearchCV(m_cfg['model'], m_cfg['params'], cv=cv, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        best_model = gs.best_estimator_
        best_t, val_f1 = find_best_threshold(best_model, X_val, y_val)
        
        y_test_probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        y_test_pred = (y_test_probs >= best_t) if y_test_probs is not None else best_model.predict(X_test)
        
        results[name] = {
            'best_params': gs.best_params_,
            'best_threshold': best_t,
            'val_f1_opt': val_f1,
            'test_f1': f1_score(y_test, y_test_pred),
            'test_acc': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_test_probs) if y_test_probs is not None else None,
            'importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else (abs(best_model.coef_[0]) if hasattr(best_model, 'coef_') else None)
        }
        print(f"      T={best_t:.2f} | Val F1: {val_f1:.4f} | Test F1: {results[name]['test_f1']:.4f}")
    
    return results

def plot_feature_importance(results, feature_names, config):
    priority = ['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest']
    best_model_name = next((m for m in priority if m in results), list(results.keys())[0])
    
    if results[best_model_name]['importance'] is not None:
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': results[best_model_name]['importance']
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 12))
        sns.barplot(x='importance', y='feature', data=imp_df.head(30))
        plt.title(f'Top 30 Feature Importance ({best_model_name})')
        plt.tight_layout()
        plt.savefig(config['paths']['plot_dir'] + 'feature_importance_extended.png')
        print(f"\n✓ Feature importance plot saved.")
        return imp_df
    return None

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(CONFIG)
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, CONFIG)
    imp_df = plot_feature_importance(results, feature_names, CONFIG)
    
    # Summary CSV
    summary_list = []
    for name, res in results.items():
        summary_list.append({
            'Model': name,
            'F1_Class1': res['test_f1'],
            'Accuracy': res['test_acc'],
            'Precision': res['test_precision'],
            'Recall': res['test_recall'],
            'AUC': res['test_auc'],
            'Best_Threshold': res['best_threshold']
        })
    pd.DataFrame(summary_list).to_csv(CONFIG['paths']['results_csv'], index=False)
    
    print("\n" + "=" * 80)
    print("FINAL EXTENDED COMPARISON (Class 1 F1-score)")
    print("=" * 80)
    print(pd.DataFrame(summary_list)[['Model', 'F1_Class1', 'Precision', 'Recall', 'Best_Threshold']].to_string(index=False))
