"""
Loan Default Risk Analysis
Phase 1 — Data Foundation
---------------------------------
Steps:
1. Load the raw CSV
2. Inspect shape, dtypes, missing values, duplicates
3. Drop LoanID (pure identifier, no predictive value)
4. Generate descriptive statistics (numeric + categorical)
5. Check target variable class balance
6. Save a cleaned version of the dataset for later phases
"""

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("../data/processed/loan_default_clean.csv")

print("=" * 70)
print("1. SHAPE")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

# ---------------------------------------------------------
# 2. DATA TYPES
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("2. COLUMN DATA TYPES")
print("=" * 70)
print(df.dtypes)

# ---------------------------------------------------------
# 3. MISSING VALUES
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("3. MISSING VALUES PER COLUMN")
print("=" * 70)
print(df.isnull().sum())

# ---------------------------------------------------------
# 4. DUPLICATES
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("4. DUPLICATE ROWS")
print("=" * 70)
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate LoanIDs: {df['LoanID'].duplicated().sum()}")

# ---------------------------------------------------------
# 5. DROP LoanID
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("5. DROP LoanID (pure identifier, no predictive value)")
print("=" * 70)
if 'LoanID' in df.columns:
    df = df.drop(columns=['LoanID'])
print(f"New shape: {df.shape}")

# ---------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS — NUMERIC COLUMNS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("6. DESCRIPTIVE STATISTICS - NUMERIC COLUMNS")
print("=" * 70)
print(df.describe().T)

# ---------------------------------------------------------
# 7. CATEGORICAL COLUMNS — VALUE COUNTS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("7. CATEGORICAL COLUMNS - VALUE COUNTS")
print("=" * 70)
cat_cols = df.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    print(f"\n--- {c} ---")
    print(df[c].value_counts())

# ---------------------------------------------------------
# 8. TARGET VARIABLE — CLASS BALANCE
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("8. TARGET VARIABLE - CLASS BALANCE")
print("=" * 70)
print(df['Default'].value_counts())
print(df['Default'].value_counts(normalize=True).round(4) * 100)

# ---------------------------------------------------------
# 9. SAVE CLEANED DATASET FOR NEXT PHASES
# ---------------------------------------------------------
df.to_csv("../data/processed/loan_default_clean.csv", index=False)
print("\nSaved cleaned dataset -> ../data/processed/loan_default_clean.csv")
