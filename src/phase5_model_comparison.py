"""
Loan Default Risk Analysis
Phase 5 — Model Comparison
--------------------------------------------------------
Trains and evaluates 5 models on the same train/test split:
1. Logistic Regression (Baseline)        - no imbalance handling, threshold 0.5
2. Logistic Regression (Class-Weighted)  - class_weight='balanced'
3. Logistic Regression (SMOTE)           - SMOTE applied to TRAINING data only
4. Random Forest (Class-Weighted)        - class_weight='balanced'
5. XGBoost (scale_pos_weight)            - built-in imbalance handling

IMPORTANT: SMOTE is fit only on the training set, never on the test set.
Applying SMOTE before the train/test split (or on the full dataset) causes
DATA LEAKAGE - synthetic points derived from test-set-like data would leak
into training, inflating recall artificially. This is a common mistake in
public examples of this exact dataset - avoided here deliberately.

Input : train_data.csv, test_data.csv (output of Phase 4)
Output: phase5_model_comparison.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              average_precision_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

train = pd.read_csv("../data/processed/train_data.csv")
test = pd.read_csv("../data/processed/test_data.csv")

X_train = train.drop(columns=['Default'])
y_train = train['Default']
X_test = test.drop(columns=['Default'])
y_test = test['Default']

results = []


def evaluate(name, model, X_tr, y_tr, X_te, y_te, threshold=0.5):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_te, pred)
    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred, zero_division=0)
    roc_auc = roc_auc_score(y_te, proba)
    pr_auc = average_precision_score(y_te, proba)

    results.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc,
        'TN': cm[0, 0], 'FP': cm[0, 1], 'FN': cm[1, 0], 'TP': cm[1, 1]
    })

    print(f"\n--- {name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} "
          f"F1={f1:.4f} ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
    return model, proba


# ===========================================================
# 1. LOGISTIC REGRESSION - BASELINE
# ===========================================================
lr_base = LogisticRegression(max_iter=1000, random_state=42)
evaluate("1. Logistic Regression (Baseline)", lr_base, X_train, y_train,
          X_test, y_test)

# ===========================================================
# 2. LOGISTIC REGRESSION - CLASS WEIGHTED
# ===========================================================
lr_weighted = LogisticRegression(max_iter=1000, class_weight='balanced',
                                   random_state=42)
evaluate("2. Logistic Regression (Class-Weighted)", lr_weighted, X_train,
          y_train, X_test, y_test)

# ===========================================================
# 3. LOGISTIC REGRESSION - SMOTE (train set only!)
# ===========================================================
print("\nApplying SMOTE to training data only...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {y_train_smote.value_counts().to_dict()}")

lr_smote = LogisticRegression(max_iter=1000, random_state=42)
evaluate("3. Logistic Regression (SMOTE)", lr_smote, X_train_smote,
          y_train_smote, X_test, y_test)

# ===========================================================
# 4. RANDOM FOREST - CLASS WEIGHTED
# ===========================================================
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                              random_state=42, n_jobs=-1)
evaluate("4. Random Forest (Class-Weighted)", rf, X_train, y_train,
          X_test, y_test)

# ===========================================================
# 5. XGBOOST - scale_pos_weight
# ===========================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(n_estimators=200,
                                scale_pos_weight=scale_pos_weight,
                                random_state=42, eval_metric='logloss')
evaluate("5. XGBoost (scale_pos_weight)", xgb_model, X_train, y_train,
          X_test, y_test)

# ===========================================================
# FINAL COMPARISON TABLE
# ===========================================================
print("\n\n" + "=" * 100)
print("FINAL COMPARISON TABLE")
print("=" * 100)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("../results/phase5_model_comparison.csv", index=False)
print("\nSaved phase5_model_comparison.csv")
