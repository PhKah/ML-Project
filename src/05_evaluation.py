import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'results_dir': 'Data/',
        'plot_dir': 'plots/'
    },
    'model': {
        'name': 'CatBoost',
        'params': {
            'n_estimators': 100,
            'random_state': 42,
            'verbose': 0,
            'auto_class_weights': 'Balanced',
            'learning_rate': 0.1,
            'depth': 6
        },
        'threshold': 0.27 # From Task 04 optimization
    }
}

print("="*80)
print("TASK 05: SYSTEMATIC EVALUATION (CATBOOST PRE-MATCH)")
print("="*80)

# ==============================================================================
# STEP 0: Load and Prepare Data
# ==============================================================================
print("\n[0] Loading data and preparing Honest Split...")
df = pd.read_csv(CONFIG['paths']['input_data'])

X = df.drop(['iid', 'pid', 'match'], axis=1)
y = df['match']
feature_names = X.columns.tolist()

# 60:20:20 Split (Must match Task 04 logic)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Apply SMOTE to Train ONLY
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ==============================================================================
# STEP 1: Fit Final Model
# ==============================================================================
print(f"\n[1] Fitting final {CONFIG['model']['name']} model...")
model = CatBoostClassifier(**CONFIG['model']['params'])
model.fit(X_train_res, y_train_res)

# ==============================================================================
# LEVEL 1: Surface Metrics (Honest Focus)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 1: SURFACE METRICS (Focus: Class 1 F1)")
print("="*80)

T = CONFIG['model']['threshold']
y_probs_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_probs_test >= T)

print(f"\n📊 Performance on TEST SET (Threshold={T}):")
print(classification_report(y_test, y_pred_test, target_names=['No Match', 'Match']))

roc_auc = roc_auc_score(y_test, y_probs_test)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ==============================================================================
# LEVEL 2: Structure (Bias-Variance via Learning Curves)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 2: STRUCTURE (Bias-Variance Diagnostic)")
print("="*80)

print("   Calculating learning curves (using F1 scoring)...")
train_sizes, train_scores, val_scores = learning_curve(
    CatBoostClassifier(**CONFIG['model']['params']), 
    X_train_res, y_train_res, cv=3, scoring='f1', 
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

gap = train_mean[-1] - val_mean[-1]
diagnosis = "✓ GOOD" if gap < 0.05 else "⚠️ OVERFITTING"
print(f"   Gap (Train - Val): {gap:.4f} -> {diagnosis}")

# ==============================================================================
# LEVEL 3: Deep Dive (Error Surgery)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 3: DEEP DIVE (Error Surgery - FP/FN Analysis)")
print("="*80)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print(f"   Confusion Matrix Results:")
print(f"   • Correct 'No Match' (TN): {tn}")
print(f"   • Correct 'Match'    (TP): {tp}")
print(f"   • Missed Matches     (FN): {fn} (DANGER: Opportunity cost)")
print(f"   • False Alarms       (FP): {fp} (NOISE: User frustration)")

# ==============================================================================
# LEVEL 4: Root Cause (Data Quality)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 4: ROOT CAUSE (Outlier Check on Misclassified)")
print("="*80)

mis_mask = (y_test != y_pred_test)
X_test_array = X_test.values
z_scores = np.abs((X_test_array - X_test_array.mean(axis=0)) / (X_test_array.std(axis=0) + 1e-8))
outlier_count = (z_scores > 3).sum(axis=1)

print(f"   • Outlier rate in CORRECT: {outlier_count[~mis_mask].mean():.2f} features/sample")
print(f"   • Outlier rate in ERRORS:  {outlier_count[mis_mask].mean():.2f} features/sample")

# ==============================================================================
# LEVEL 5: Operations (Fairness - Gender)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 5: OPERATIONS (Fairness Check - Gender)")
print("="*80)

genders = X_test['gender'].unique()
for g in genders:
    mask = (X_test['gender'] == g)
    label = "Male" if g == 1 else "Female"
    f1_g = f1_score(y_test[mask], y_pred_test[mask])
    print(f"   • {label:6s} (N={mask.sum()}): F1-score = {f1_g:.4f}")

# ==============================================================================
# PLOTTING & SAVING
# ==============================================================================
print("\n[2] Generating 5-Level Diagnostics Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Diagnostic Dashboard: {CONFIG["model"]["name"]} (Pre-match)', fontsize=16)

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0,0])
axes[0,0].set_title('Level 3: Confusion Matrix')
axes[0,0].set_ylabel('True')
axes[0,0].set_xlabel('Predicted')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs_test)
axes[0,1].plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
axes[0,1].plot([0, 1], [0, 1], 'k--')
axes[0,1].set_title('Level 1: ROC Curve')
axes[0,1].legend()

# 3. Learning Curves
axes[1,0].plot(train_sizes, train_mean, 'o-', label='Train F1')
axes[1,0].plot(train_sizes, val_mean, 's-', label='Val F1')
axes[1,0].set_title('Level 2: Learning Curves')
axes[1,0].legend()

# 4. Feature Importance (Top 15)
imp = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
sns.barplot(x='importance', y='feature', data=imp.head(15), ax=axes[1,1])
axes[1,1].set_title('Level 5: Top 15 Features')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(CONFIG['paths']['plot_dir'] + 'evaluation_diagnostics.png')

print(f"\n✓ Dashboard saved to {CONFIG['paths']['plot_dir']}evaluation_diagnostics.png")
print("\n" + "="*80)
print("✓ TASK 05 COMPLETED")
print("="*80)
