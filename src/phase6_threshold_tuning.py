"""
Loan Default Risk Analysis
Phase 6 — Threshold Tuning + ROC/PR Curves
--------------------------------------------------------
Steps:
1. Plot ROC curves for the 3 strongest models from Phase 5
2. Plot Precision-Recall curves (more informative than ROC on
   imbalanced data - see PR-AUC discussion from earlier phases)
3. Find the F1-optimal classification threshold for each model
   instead of blindly using the default 0.5 cutoff

Input : train_data.csv, test_data.csv (Phase 4 output)
Output: phase6_roc_curves.png, phase6_pr_curves.png,
        phase6_threshold_tuning.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              average_precision_score, precision_score,
                              recall_score, f1_score)
import xgboost as xgb

train = pd.read_csv("../data/processed/train_data.csv")
test = pd.read_csv("../data/processed/test_data.csv")
X_train = train.drop(columns=['Default'])
y_train = train['Default']
X_test = test.drop(columns=['Default'])
y_test = test['Default']

# ===========================================================
# Retrain the 3 strongest models from Phase 5 for tuning
# ===========================================================
models = {}

lr_w = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_w.fit(X_train, y_train)
models['Logistic Regression (Class-Weighted)'] = lr_w.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                              random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['Random Forest (Class-Weighted)'] = rf.predict_proba(X_test)[:, 1]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_m = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight,
                            random_state=42, eval_metric='logloss')
xgb_m.fit(X_train, y_train)
models['XGBoost'] = xgb_m.predict_proba(X_test)[:, 1]

# ===========================================================
# 1. ROC CURVES
# ===========================================================
plt.figure(figsize=(8, 7))
for name, proba in models.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/phase6_roc_curves.png', dpi=110)
plt.close()

# ===========================================================
# 2. PRECISION-RECALL CURVES
# ===========================================================
plt.figure(figsize=(8, 7))
for name, proba in models.items():
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.plot(rec, prec, label=f'{name} (PR-AUC={ap:.3f})')
baseline = y_test.mean()
plt.axhline(baseline, color='k', linestyle='--',
            label=f'Baseline (default rate={baseline:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/phase6_pr_curves.png', dpi=110)
plt.close()

# ===========================================================
# 3. THRESHOLD TUNING - find F1-optimal cutoff per model
# ===========================================================
print("=" * 90)
print("THRESHOLD TUNING - finding F1-optimal cutoff per model")
print("=" * 90)

tuning_results = []
for name, proba in models.items():
    prec, rec, thresh = precision_recall_curve(y_test, proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no threshold
    best_thresh = thresh[best_idx]

    pred_05 = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= best_thresh).astype(int)

    print(f"\n--- {name} ---")
    print(f"Optimal threshold (max F1) = {best_thresh:.4f}")
    print(f"  At threshold=0.5   : "
          f"Precision={precision_score(y_test, pred_05, zero_division=0):.4f} "
          f"Recall={recall_score(y_test, pred_05, zero_division=0):.4f} "
          f"F1={f1_score(y_test, pred_05, zero_division=0):.4f}")
    print(f"  At tuned threshold : "
          f"Precision={precision_score(y_test, pred_tuned, zero_division=0):.4f} "
          f"Recall={recall_score(y_test, pred_tuned, zero_division=0):.4f} "
          f"F1={f1_score(y_test, pred_tuned, zero_division=0):.4f}")

    tuning_results.append({
        'Model': name, 'Optimal_Threshold': best_thresh,
        'Recall_at_0.5': recall_score(y_test, pred_05, zero_division=0),
        'Recall_at_tuned': recall_score(y_test, pred_tuned, zero_division=0),
        'F1_at_0.5': f1_score(y_test, pred_05, zero_division=0),
        'F1_at_tuned': f1_score(y_test, pred_tuned, zero_division=0),
        'Precision_at_tuned': precision_score(y_test, pred_tuned, zero_division=0)
    })

tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv("../results/phase6_threshold_tuning.csv", index=False)
print("\n\nSaved phase6_threshold_tuning.csv, phase6_roc_curves.png, "
      "phase6_pr_curves.png")
