"""
Loan Default Risk Analysis
Phase 2 — Exploratory Data Analysis (EDA)
---------------------------------------------
Steps:
1. Univariate analysis: histograms + skewness/kurtosis for numeric variables
2. Bivariate analysis (numeric): boxplots of each numeric variable vs Default
3. Bivariate analysis (categorical): default rate % by category
4. Correlation matrix heatmap (numeric variables + Default)

Input : loan_default_clean.csv (output of Phase 1)
Output: 3 PNG charts + printed summary tables
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # remove this line if running interactively / in Jupyter
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/loan_default_clean.csv")

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
            'HasDependents', 'LoanPurpose', 'HasCoSigner']

# ===========================================================
# 1. UNIVARIATE ANALYSIS — histograms with skew/kurtosis
# ===========================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

print("SKEW / KURTOSIS TABLE")
print("=" * 50)
for i, col in enumerate(numeric_cols):
    vals = df[col].values
    mu, sigma = np.mean(vals), np.std(vals)
    skew = np.mean(((vals - mu) / sigma) ** 3)
    kurt = np.mean(((vals - mu) / sigma) ** 4) - 3
    print(f"{col:18s} skew={skew:7.3f}  kurtosis={kurt:7.3f}")

    axes[i].hist(vals, bins=30, density=True, edgecolor='black', alpha=0.7)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axes[i].plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='orange')
    axes[i].set_title(f'{col}\nSkew={skew:.2f}, Kurt={kurt:.2f}', fontsize=10)

plt.tight_layout()
plt.savefig('../figures/phase2_univariate_histograms.png', dpi=110)
plt.close()

# ===========================================================
# 2. BIVARIATE ANALYSIS — numeric vars vs Default (boxplots)
# ===========================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

print("\nMEDIAN COMPARISON: Default vs Non-Default")
print("=" * 60)
for i, col in enumerate(numeric_cols):
    default = df[df['Default'] == 1][col]
    non_default = df[df['Default'] == 0][col]
    med_d, med_nd = default.median(), non_default.median()
    print(f"{col:18s} Non-Default median={med_nd:>10.2f}   Default median={med_d:>10.2f}")

    axes[i].boxplot([non_default, default], tick_labels=['Non-Default', 'Default'])
    axes[i].set_title(col, fontsize=10)

plt.tight_layout()
plt.savefig('../figures/phase2_boxplots_vs_default.png', dpi=110)
plt.close()

# ===========================================================
# 3. BIVARIATE ANALYSIS — categorical vars vs Default (rate %)
# ===========================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
overall_rate = df['Default'].mean() * 100

print(f"\nDEFAULT RATE % BY CATEGORY (overall avg = {overall_rate:.2f}%)")
print("=" * 60)
for i, col in enumerate(cat_cols):
    rate = df.groupby(col)['Default'].mean().sort_values(ascending=False) * 100
    print(f"\n--- {col} ---")
    print(rate.round(2))

    axes[i].bar(rate.index.astype(str), rate.values, color='#d95f5f')
    axes[i].set_title(f'Default Rate % by {col}', fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].axhline(overall_rate, color='black', linestyle='--', linewidth=1)

for j in range(len(cat_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('../figures/phase2_default_rate_by_category.png', dpi=110)
plt.close()

# ===========================================================
# 4. CORRELATION MATRIX HEATMAP
# ===========================================================
corr_cols = numeric_cols + ['Default']
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90)
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('../figures/phase2_correlation_heatmap.png', dpi=110)
plt.close()

print("\nCORRELATION WITH DEFAULT (sorted by absolute strength)")
print("=" * 55)
print(corr['Default'].drop('Default').sort_values(key=abs, ascending=False))

print("\nSaved charts: phase2_univariate_histograms.png, "
      "phase2_boxplots_vs_default.png, "
      "phase2_default_rate_by_category.png, "
      "phase2_correlation_heatmap.png")
