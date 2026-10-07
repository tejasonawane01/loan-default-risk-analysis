"""
Loan Default Risk Analysis
Phase 4 — Model-Ready Data Preparation
------------------------------------------
Steps:
1. One-hot encode all categorical variables (fixes the original report's
   biggest flaw: chi-square proved these mattered, but the original
   logistic regression model dropped them entirely)
2. Check multicollinearity among numeric predictors via VIF
3. Stratified 70/30 train/test split — preserves the ~11.6% default
   rate in both sets, so evaluation isn't skewed by an unlucky split

Input : loan_default_clean.csv (output of Phase 1)
Output: train_data.csv, test_data.csv (used in Phase 5 modeling)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("../data/processed/loan_default_clean.csv")

cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
            'HasDependents', 'LoanPurpose', 'HasCoSigner']
num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# ===========================================================
# 1. ONE-HOT ENCODING
# ===========================================================
print("=" * 70)
print("1. ONE-HOT ENCODING")
print("=" * 70)
print(f"Before encoding: {df.shape[1]} columns")

# drop_first=True avoids the "dummy variable trap" (perfect multicollinearity
# between dummy columns of the same original variable)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print(f"After encoding: {df_encoded.shape[1]} columns")
new_cols = [c for c in df_encoded.columns if c not in df.columns]
print("\nNew columns added:")
for c in new_cols:
    print(f"  {c}")

# ===========================================================
# 2. VIF CHECK (multicollinearity among numeric predictors)
# ===========================================================
print("\n" + "=" * 70)
print("2. VIF CHECK (multicollinearity among numeric predictors)")
print("=" * 70)
X_num = df[num_cols].copy()
X_num = X_num.assign(const=1)  # VIF calculation requires a constant term
vif_data = pd.DataFrame()
vif_data["Variable"] = num_cols
vif_data["VIF"] = [variance_inflation_factor(X_num.values, i)
                    for i in range(len(num_cols))]
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))
print("\nGuide: VIF < 5 = no concerning multicollinearity | "
      "5-10 = moderate | >10 = problematic")

# ===========================================================
# 3. STRATIFIED TRAIN/TEST SPLIT
# ===========================================================
print("\n" + "=" * 70)
print("3. STRATIFIED TRAIN/TEST SPLIT (70/30)")
print("=" * 70)
X = df_encoded.drop(columns=['Default'])
y = df_encoded['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"\nTrain default rate: {y_train.mean() * 100:.4f}%")
print(f"Test default rate:  {y_test.mean() * 100:.4f}%")
print(f"Full data default rate: {y.mean() * 100:.4f}%")

# Save for Phase 5 modeling
train = X_train.copy()
train['Default'] = y_train
test = X_test.copy()
test['Default'] = y_test
train.to_csv("../data/processed/train_data.csv", index=False)
test.to_csv("../data/processed/test_data.csv", index=False)
print("\nSaved train_data.csv and test_data.csv for Phase 5")
