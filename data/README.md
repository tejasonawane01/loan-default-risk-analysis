# Data

This directory contains the datasets used in the Loan Default Risk Analysis project.

## Directory Structure

```
data/
├── raw/                    # Original, unmodified datasets
│   ├── german_credit_raw.data   # UCI Statlog German Credit Data (1000 records)
│   └── german_credit_clean.csv  # Cleaned version of the German Credit data
└── processed/              # Cleaned and transformed datasets
    ├── loan_default_clean.csv   # Main cleaned dataset (255,347 records, 17 features)
    ├── train_data.csv           # Training set (70% split, one-hot encoded)
    └── test_data.csv            # Test set (30% split, one-hot encoded)
```

## Dataset Descriptions

### Main Dataset: `loan_default_clean.csv`

- **Source**: Kaggle Loan Default Dataset
- **Records**: 255,347 loan records
- **Features**: 17 variables (9 numeric + 7 categorical + 1 target)
- **Target**: `Default` (0 = No Default, 1 = Default)
- **Default Rate**: ~11.6%

**Numeric Features:**
- `Age`: Borrower's age
- `Income`: Annual income
- `LoanAmount`: Loan amount requested
- `CreditScore`: Credit score
- `MonthsEmployed`: Months employed
- `NumCreditLines`: Number of credit lines
- `InterestRate`: Interest rate on the loan
- `LoanTerm`: Loan term in months
- `DTIRatio`: Debt-to-income ratio

**Categorical Features:**
- `Education`: Education level
- `EmploymentType`: Type of employment
- `MaritalStatus`: Marital status
- `HasMortgage`: Whether the borrower has a mortgage
- `HasDependents`: Whether the borrower has dependents
- `LoanPurpose`: Purpose of the loan
- `HasCoSigner`: Whether the loan has a co-signer

### German Credit Dataset: `german_credit_raw.data`

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Citation**: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset].
- **Records**: 1,000 loan applicants
- **Features**: 20 variables (7 numeric + 13 categorical + 1 target)
- **Target**: `Default` (0 = Good credit, 1 = Bad credit)
- **Default Rate**: 30%

This dataset is used in Phase 3B as a real-world validation benchmark to compare effect sizes against the main dataset.

## Data Processing Notes

1. The main dataset was cleaned by dropping the `LoanID` column (pure identifier with no predictive value).
2. Categorical variables were one-hot encoded for modeling (Phase 4).
3. A stratified 70/30 train/test split was used to preserve the default rate in both sets.
4. The German Credit dataset was recoded to match the main dataset's convention (0 = no default, 1 = default).

## License and Redistribution

- **Main Dataset**: The Kaggle Loan Default dataset is assumed to be publicly available for educational and research purposes. Please refer to Kaggle's terms of use for redistribution rights.
- **German Credit Dataset**: The UCI Statlog German Credit Data is publicly available from the UCI Machine Learning Repository for research and educational use.

## Missing Data

The original raw Excel file (`Loan_default.xlsx`) is not included in this repository due to its large size (~20 MB). The cleaned CSV version is provided instead. If you need to reproduce the cleaning process from the original Excel file, please download it from Kaggle and update the path in `src/phase1_data_foundation.py`.
