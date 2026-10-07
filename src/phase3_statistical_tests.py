"""
Loan Default Risk Analysis
Phase 3 — Statistical Hypothesis Testing (with Effect Sizes)
-----------------------------------------------------------------
Steps:
1. Chi-Square Test of Independence for each categorical variable vs Default
   + Cramer's V (effect size for chi-square)
2. Mann-Whitney U Test for each numeric variable vs Default
   + Rank-biserial correlation (effect size for Mann-Whitney)

Why effect sizes matter:
With a very large sample size (255,347 rows), almost ANY difference
becomes "statistically significant" (p < 0.05), even if it is tiny and
practically meaningless. Effect size tells us how BIG the relationship
actually is, separate from whether it is statistically real.

Input : loan_default_clean.csv (output of Phase 1)
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("../data/processed/loan_default_clean.csv")

cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
            'HasDependents', 'LoanPurpose', 'HasCoSigner']
num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'InterestRate']


# ===========================================================
# 1. CHI-SQUARE TESTS + CRAMER'S V
# ===========================================================
def cramers_v(chi2, n, r, c):
    """Effect size for chi-square test. 0 = no association, 1 = perfect association."""
    return np.sqrt((chi2 / n) / (min(r - 1, c - 1)))


print("=" * 75)
print("CHI-SQUARE TESTS (categorical vs Default) + CRAMER'S V")
print("=" * 75)

for col in cat_cols:
    table = pd.crosstab(df[col], df['Default'])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.sum().sum()
    r, c = table.shape
    v = cramers_v(chi2, n, r, c)
    sig = "YES" if p < 0.05 else "no"
    print(f"{col:16s}  Chi2={chi2:9.2f}  df={dof}  p={p:.6f}  "
          f"Cramer'sV={v:.4f}  Significant={sig}")

print("\nCramer's V guide: <0.10 negligible | 0.10-0.20 weak | "
      "0.20-0.40 moderate | >0.40 strong")

# ===========================================================
# 2. MANN-WHITNEY U TESTS + RANK-BISERIAL CORRELATION
# ===========================================================
print("\n" + "=" * 75)
print("MANN-WHITNEY U TESTS (numeric vs Default) + RANK-BISERIAL CORRELATION")
print("=" * 75)

for col in num_cols:
    d = df[df['Default'] == 1][col]
    nd = df[df['Default'] == 0][col]
    u_stat, p = stats.mannwhitneyu(d, nd, alternative='two-sided')
    n1, n2 = len(d), len(nd)
    rb = 1 - (2 * u_stat) / (n1 * n2)  # rank-biserial correlation
    sig = "YES" if p < 0.05 else "no"
    print(f"{col:14s}  U={u_stat:15.1f}  p={p:.6e}  "
          f"rank-biserial={rb:7.4f}  Significant={sig}")

print("\nRank-biserial guide: same -1 to 1 scale as correlation; "
      "interpreted with the same strength bands as Cramer's V")
