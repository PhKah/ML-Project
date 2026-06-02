import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay

# 1. Load data
df = pd.read_csv('Data/data_final.csv')
X = df.drop('match', axis=1)
y = df['match']

# 2. Re-split to get the exact same test set (using same random_state as Task 04)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 3. Re-train the best model (Decision Tree with max_depth=15, min_samples_split=2)
# In a real project, we would load the saved model, but here we re-train for simplicity.
best_dt = DecisionTreeClassifier(max_depth=15, min_samples_split=2, random_state=42)
best_dt.fit(X_train, y_train)

# 4. Evaluation on Test Set
y_test_pred = best_dt.predict(X_test)
y_test_proba = best_dt.predict_proba(X_test)[:, 1]

print("--- Final Evaluation on Test Set ---")
print(classification_report(y_test, y_test_pred))

# 5. Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix - Test Set')
plt.savefig('plots/05_confusion_matrix.png')
print("\nConfusion Matrix saved to plots/05_confusion_matrix.png")

# 6. ROC Curve Visualization
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_dt, X_test, y_test, ax=ax)
plt.title('ROC Curve - Test Set')
plt.savefig('plots/05_roc_curve.png')
print("ROC Curve saved to plots/05_roc_curve.png")

# 7. Feature Importance Analysis
importances = best_dt.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
top_10_features = feat_importances.nlargest(10)

fig, ax = plt.subplots(figsize=(10, 8))
top_10_features.sort_values().plot(kind='barh', color='skyblue', ax=ax)
plt.title('Top 10 Most Important Features (Decision Tree)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/05_feature_importance.png')
print("Feature Importance plot saved to plots/05_feature_importance.png")

print("\n--- Top 10 Features ---")
print(top_10_features)

