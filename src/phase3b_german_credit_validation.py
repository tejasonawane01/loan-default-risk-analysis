"""
Loan Default Risk Analysis
Phase 3B — Real-World Validation Benchmark (German Credit Dataset)
-------------------------------------------------------------------
Purpose:
Our main dataset (Kaggle Loan Default, 255,347 rows) shows statistical
signatures consistent with synthetic generation (near-uniform numeric
distributions, evenly-split categories, negligible effect sizes even
for theoretically important variables like Credit Score and DTI).

To test this, we run the EXACT same Chi-Square / Mann-Whitney pipeline
(Phase 3) on a small, genuinely real-world credit dataset: the UCI
Statlog German Credit Data (Hofmann, 1994) - 1000 real loan applicants,
classified as Good or Bad credit risk.

If categorical/numeric variables show noticeably STRONGER effect sizes
here than in our synthetic dataset, that is direct evidence our main
dataset does not reflect real-world credit risk behavior.

Source: UCI Machine Learning Repository
        https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
        Hofmann, H. (1994). Statlog (German Credit Data) [Dataset].
"""

import pandas as pd
import numpy as np
from scipy import stats

# ===========================================================
# 1. LOAD + CLEAN RAW UCI DATA
# ===========================================================
cols = ["chk_acct", "duration", "credit_history", "purpose", "credit_amount",
        "savings_acct", "present_emp", "installment_rate", "personal_status_sex",
        "other_debtor", "present_resid", "property", "age", "other_install",
        "housing", "num_credits", "job", "num_dependents", "telephone",
        "foreign_worker", "response"]

df = pd.read_csv("../data/raw/german_credit_raw.data", sep=';', header=None, names=cols,
                  encoding='utf-8-sig')

# response: 1 = Good credit (no default), 2 = Bad credit (default)
# Recode to match our Default convention: 0 = no default, 1 = default
df['Default'] = df['response'].map({1: 0, 2: 1})
df = df.drop(columns=['response'])

print(f"Shape: {df.shape}")
print(f"Default rate: {df['Default'].mean()*100:.2f}%  "
      f"(Good={  (df['Default']==0).sum()}, Bad={(df['Default']==1).sum()})")

df.to_csv("../data/raw/german_credit_clean.csv", index=False)


# ===========================================================
# 2. CHI-SQUARE TESTS + CRAMER'S V (categorical vs Default)
# ===========================================================
def cramers_v(chi2, n, r, c):
    return np.sqrt((chi2 / n) / (min(r - 1, c - 1)))


print("\n" + "=" * 75)
print("GERMAN CREDIT - CHI-SQUARE TESTS + CRAMER'S V")
print("=" * 75)

cat_cols = ['personal_status_sex', 'present_emp', 'housing', 'purpose',
            'property', 'job']
for col in cat_cols:
    table = pd.crosstab(df[col], df['Default'])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.sum().sum()
    r, c = table.shape
    v = cramers_v(chi2, n, r, c)
    sig = "YES" if p < 0.05 else "no"
    print(f"{col:20s} Chi2={chi2:8.2f}  df={dof}  p={p:.6f}  "
          f"Cramer'sV={v:.4f}  Significant={sig}")


# ===========================================================
# 3. MANN-WHITNEY U TESTS + RANK-BISERIAL (numeric vs Default)
# ===========================================================
print("\n" + "=" * 75)
print("GERMAN CREDIT - MANN-WHITNEY U TESTS + RANK-BISERIAL CORRELATION")
print("=" * 75)

num_cols = ['age', 'credit_amount', 'duration', 'installment_rate']
for col in num_cols:
    d = df[df['Default'] == 1][col]
    nd = df[df['Default'] == 0][col]
    u, p = stats.mannwhitneyu(d, nd, alternative='two-sided')
    n1, n2 = len(d), len(nd)
    rb = 1 - (2 * u) / (n1 * n2)
    sig = "YES" if p < 0.05 else "no"
    print(f"{col:18s} U={u:12.1f}  p={p:.6e}  "
          f"rank-biserial={rb:7.4f}  Significant={sig}")

print("\nCompare these Cramer's V / rank-biserial values against Phase 3's "
      "results on the Kaggle loan dataset — real data should show "
      "noticeably larger effect sizes.")
