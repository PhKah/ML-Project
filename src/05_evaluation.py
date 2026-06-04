import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, fbeta_score
)
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
        'test_data': 'Data/test_set_hidden.csv',
        'winner_model': 'models/winner_model.joblib',
        'plot_dir': 'plots/'
    }
}

print("="*80)
print("TASK 05: [REFAC] FINAL EVALUATION (F-BETA 0.5 & PRECISION FOCUS)")
print("="*80)

# ==============================================================================
# STEP 0: Load Model and Data
# ==============================================================================
print("\n[0] Loading Locked Winner Model and Hidden Test Set...")

if not os.path.exists(CONFIG['paths']['winner_model']):
    raise FileNotFoundError(f"Winner model not found at {CONFIG['paths']['winner_model']}. Run Task 04 first.")

# Load winner info
winner = joblib.load(CONFIG['paths']['winner_model'])
model_name = winner['name']
pipeline = winner['pipeline']
threshold = winner['threshold']

print(f"   ✓ Locked Winner: {model_name}")
print(f"   ✓ Optimal Threshold: {threshold:.2f}")

# Load test data
test_df = pd.read_csv(CONFIG['paths']['test_data'])
X_test = test_df.drop('match', axis=1)
y_test = test_df['match']
feature_names = X_test.columns.tolist()

print(f"   ✓ Test samples: {len(X_test)}")

# ==============================================================================
# LEVEL 1: Surface Metrics (F-beta Focus)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 1: SURFACE METRICS (Final Performance)")
print("="*80)

if hasattr(pipeline, "predict_proba"):
    y_probs_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= threshold)
else:
    y_pred_test = pipeline.predict(X_test)
    y_probs_test = None

print(f"\n📊 FINAL PERFORMANCE ON TEST SET ({model_name}):")
print(classification_report(y_test, y_pred_test, target_names=['No Match', 'Match']))

f1 = f1_score(y_test, y_pred_test)
f05 = fbeta_score(y_test, y_pred_test, beta=0.5, zero_division=0)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test)

print(f"📈 STRATEGIC METRICS:")
print(f"   • F1-Score:     {f1:.4f}")
print(f"   • F0.5-Score:   {f05:.4f} (Precision weight x2)")
print(f"   • Precision:    {prec:.4f}")
print(f"   • Recall:       {rec:.4f}")

if y_probs_test is not None:
    roc_auc = roc_auc_score(y_test, y_probs_test)
    print(f"   • ROC-AUC:      {roc_auc:.4f}")

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
print(f"   • False Alarms       (FP): {fp} (NOISE: User frustration - MINIMIZED)")

# ==============================================================================
# LEVEL 5: Operations (Fairness - Gender)
# ==============================================================================
print("\n" + "="*80)
print("LEVEL 5: OPERATIONS (Fairness Check - Gender)")
print("="*80)

if 'gender' in X_test.columns:
    genders = X_test['gender'].unique()
    for g in genders:
        mask = (X_test['gender'] == g)
        label = "Male" if g == 1 else "Female"
        f1_g = f1_score(y_test[mask], y_pred_test[mask])
        f05_g = fbeta_score(y_test[mask], y_pred_test[mask], beta=0.5, zero_division=0)
        print(f"   • {label:6s} (N={mask.sum():<4}): F1 = {f1_g:.4f} | F0.5 = {f05_g:.4f}")

# ==============================================================================
# PLOTTING & SAVING
# ==============================================================================
print("\n[2] Generating 5-Level Diagnostics Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Final Diagnostic Dashboard: {model_name} (Knowledge-Enriched & Precision-First)', fontsize=16)

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0,0])
axes[0,0].set_title('Level 3: Confusion Matrix')
axes[0,0].set_ylabel('True')
axes[0,0].set_xlabel('Predicted')

# 2. ROC Curve
if y_probs_test is not None:
    fpr, tpr, _ = roc_curve(y_test, y_probs_test)
    axes[0,1].plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_title('Level 1: ROC Curve')
    axes[0,1].legend()

# 3. Feature Importance (Top 15)
inner_model = pipeline.named_steps['model']
if hasattr(inner_model, 'feature_importances_'):
    imp = pd.DataFrame({'feature': feature_names, 'importance': inner_model.feature_importances_}).sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=imp.head(15), ax=axes[1,0])
    axes[1,0].set_title('Level 5: Top 15 Features')
elif hasattr(inner_model, 'coef_'):
    imp = pd.DataFrame({'feature': feature_names, 'importance': np.abs(inner_model.coef_[0])}).sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=imp.head(15), ax=axes[1,0])
    axes[1,0].set_title('Level 5: Top 15 Coefficients')

# 4. Strategic Summary
axes[1,1].axis('off')
summary_text = (
    f"Winner Model: {model_name}\n"
    f"Optimal T: {threshold:.2f}\n\n"
    f"Test F1: {f1:.4f}\n"
    f"Test F0.5: {f05:.4f}\n"
    f"Test Precision: {prec:.4f}\n"
    f"Test Recall: {rec:.4f}"
)
axes[1,1].text(0.1, 0.4, summary_text, fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(CONFIG['paths']['plot_dir'] + 'evaluation_diagnostics_final.png')

print(f"\n✓ Dashboard saved to {CONFIG['paths']['plot_dir']}evaluation_diagnostics_final.png")
print("\n" + "="*80)
print("✓ TASK 05 COMPLETED - FINAL DECODING SUCCESSFUL")
print("="*80)
