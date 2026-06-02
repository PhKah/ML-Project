import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# 1. Load data - Relative to project root
df = pd.read_csv('Data/data_final.csv')
X = df.drop('match', axis=1)
y = df['match']

# 2. Split Data (6:2:2)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

results = {}

# 3. Baseline: Logistic Regression
print("\n--- Training Logistic Regression (Baseline) ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_val_pred = lr.predict(X_val)
f1 = f1_score(y_val, y_val_pred)
results['Logistic Regression'] = f1
print(f"F1-score (Validation): {f1:.4f}")

# 4. Decision Tree
print("\n--- Training Decision Tree ---")
dt_params = {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, scoring='f1', cv=3)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
y_val_pred_dt = best_dt.predict(X_val)
f1_dt = f1_score(y_val, y_val_pred_dt)
results['Decision Tree'] = f1_dt
print(f"Best Params: {dt_grid.best_params_}")
print(f"F1-score (Validation): {f1_dt:.4f}")

# 5. Random Forest
print("\n--- Training Random Forest ---")
rf_params = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='f1', cv=3)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_val_pred_rf = best_rf.predict(X_val)
f1_rf = f1_score(y_val, y_val_pred_rf)
results['Random Forest'] = f1_rf
print(f"Best Params: {rf_grid.best_params_}")
print(f"F1-score (Validation): {f1_rf:.4f}")

# 6. SVM
print("\n--- Training SVM ---")
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, scoring='f1', cv=3)
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
y_val_pred_svm = best_svm.predict(X_val)
f1_svm = f1_score(y_val, y_val_pred_svm)
results['SVM'] = f1_svm
print(f"Best Params: {svm_grid.best_params_}")
print(f"F1-score (Validation): {f1_svm:.4f}")

# 7. Summary
print("\n--- Summary of F1-scores on Validation Set ---")
for model, score in results.items():
    print(f"{model}: {score:.4f}")

# Find best model
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

