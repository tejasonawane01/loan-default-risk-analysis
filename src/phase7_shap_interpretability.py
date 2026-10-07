"""
Loan Default Risk Analysis
Phase 7 — Model Interpretability (SHAP)
--------------------------------------------------------
Explains WHY the model makes the predictions it does, using SHAP
(SHapley Additive exPlanations) - breaking down how much each
variable pushes a prediction up or down from the baseline.

NOTE ON MODEL USED HERE:
The full Phase 5/6 Random Forest (200 trees, unrestricted depth,
average depth ~35) is too computationally expensive for exact SHAP
TreeExplainer computation in reasonable time. This is a well-known,
standard practical constraint - SHAP's exact tree algorithm cost
grows sharply with tree depth.

The standard solution (used here) is to train a second, shallower
Random Forest (max_depth=8) specifically for interpretability
purposes. Its raw performance numbers differ slightly from the
Phase 5/6 "production" model - this is expected and fine, because
the goal here is understanding WHICH features drive risk, not
picking a final deployment model (that was already decided in
Phase 5/6).

Input : train_data.csv, test_data.csv (Phase 4 output)
Output: phase7_shap_importance.csv, phase7_shap_summary.png,
        phase7_shap_bar.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
import shap

train = pd.read_csv("../data/processed/train_data.csv")
test = pd.read_csv("../data/processed/test_data.csv")
X_train = train.drop(columns=['Default'])
y_train = train['Default']
X_test = test.drop(columns=['Default'])
y_test = test['Default']

# ===========================================================
# Train a depth-capped Random Forest for tractable SHAP computation
# ===========================================================
rf_interp = RandomForestClassifier(n_estimators=150, max_depth=8,
                                     class_weight='balanced',
                                     random_state=42, n_jobs=-1)
rf_interp.fit(X_train, y_train)

# ===========================================================
# Compute SHAP values on a sample of the test set
# ===========================================================
X_sample = X_test.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(rf_interp)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# Handle SHAP's output shape for binary classification
if isinstance(shap_values, list):
    sv = shap_values[1]  # class 1 = Default
elif np.array(shap_values).ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

# ===========================================================
# Global feature importance (mean absolute SHAP value)
# ===========================================================
mean_abs_shap = np.abs(sv).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("TOP FEATURE IMPORTANCE (by mean |SHAP value|):")
print(importance_df.to_string(index=False))
importance_df.to_csv("../results/phase7_shap_importance.csv", index=False)

# ===========================================================
# SHAP summary plots
# ===========================================================
plt.figure()
shap.summary_plot(sv, X_sample, show=False)
plt.tight_layout()
plt.savefig("../figures/phase7_shap_summary.png", dpi=110, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(sv, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig("../figures/phase7_shap_bar.png", dpi=110, bbox_inches='tight')
plt.close()

print("\nSaved phase7_shap_importance.csv, phase7_shap_summary.png, "
      "phase7_shap_bar.png")
