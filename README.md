# Loan Default Risk Analysis

**Leveraging Statistics for Smarter Lending Decisions**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project analyzes loan default risk using statistical methods and predictive modeling to identify borrower and loan characteristics associated with default. The analysis is based on a dataset of **255,347 loan records** with **18 variables** covering demographic, financial, and loan-related characteristics.

The project follows a structured 7-phase pipeline — from data foundation and exploratory analysis through statistical hypothesis testing, predictive modeling, threshold tuning, and model interpretability — providing a comprehensive framework for credit-risk assessment.

## Objectives

- **Analyze borrower characteristics** such as age, income, credit score, and employment type
- **Identify significant predictors** of loan default using statistical hypothesis testing
- **Develop a predictive model** for borrower default classification
- **Provide insights** that can support credit-risk assessment and lending decisions

## Methodology

The project follows a 7-phase pipeline:

| Phase | Description | Script |
|-------|-------------|--------|
| 1 | Data Foundation / Data Preparation | `src/phase1_data_foundation.py` |
| 2 | Exploratory Data Analysis (EDA) | `src/phase2_eda.py` |
| 3 | Statistical Hypothesis Testing | `src/phase3_statistical_tests.py` |
| 3B | Real-World Validation Benchmark | `src/phase3b_german_credit_validation.py` |
| 4 | Model-Ready Data Preparation | `src/phase4_data_prep.py` |
| 5 | Model Comparison | `src/phase5_model_comparison.py` |
| 6 | Threshold Tuning + ROC/PR Curves | `src/phase6_threshold_tuning.py` |
| 7 | Model Interpretability (SHAP) | `src/phase7_shap_interpretability.py` |

A combined script (`src/main.py`) runs all phases sequentially.

## Exploratory Data Analysis

The EDA phase (`src/phase2_eda.py`) performs the following analyses:

- **Univariate analysis**: Histograms with skewness and kurtosis for all numeric variables
- **Distribution analysis**: Visual assessment of normality via overlaid normal curves
- **Boxplots**: Distribution comparison between defaulters and non-defaulters for each numeric variable
- **Employment type vs default**: Default rate by employment type and other categorical variables
- **Correlation matrix**: Heatmap of Pearson correlations among numeric variables and the target
- **Default-risk relationships**: Bar charts showing default rate % by category for all categorical variables

## Statistical Analysis

The statistical analysis phase (`src/phase3_statistical_tests.py`) applies:

- **Chi-square tests of independence**: Tests whether each categorical variable is associated with default
- **Cramer's V**: Effect size measure for chi-square tests (0 = no association, 1 = perfect association)
- **Mann-Whitney U tests**: Non-parametric tests comparing numeric variable distributions between defaulters and non-defaulters
- **Rank-biserial correlation**: Effect size for Mann-Whitney tests

**Why effect sizes matter**: With a very large sample size (255,347 rows), almost any difference becomes statistically significant (p < 0.05), even if it is tiny and practically meaningless. Effect sizes tell us how *big* the relationship actually is, separate from whether it is statistically real.

### German Credit Validation (Phase 3B)

The main dataset shows statistical signatures consistent with synthetic generation (near-uniform numeric distributions, evenly-split categories, negligible effect sizes). To test this, the same pipeline is run on the UCI Statlog German Credit Data (1,000 real loan applicants) as a real-world benchmark.

## Predictive Modeling

The following models are implemented in `src/phase5_model_comparison.py`:

| Model | Description | Imbalance Handling |
|-------|-------------|-------------------|
| Logistic Regression (Baseline) | Standard logistic regression | None |
| Logistic Regression (Class-Weighted) | Logistic regression with class weights | `class_weight='balanced'` |
| Logistic Regression (SMOTE) | Logistic regression on SMOTE-resampled data | SMOTE on training set only |
| Random Forest (Class-Weighted) | Random forest with class weights | `class_weight='balanced'` |
| XGBoost | Gradient boosting with scale_pos_weight | `scale_pos_weight` |

**Important**: SMOTE is fit only on the training set, never on the test set, to avoid data leakage.

## Logistic Regression Results

The baseline logistic regression model achieved the following on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | 88.49% |
| Precision (default) | 57.30% |
| Recall (default) | 3.44% |
| F1-score (default) | 6.49% |
| ROC-AUC | 0.747 |
| PR-AUC | 0.302 |

> **Note**: The high accuracy is misleading. The model has very low recall for actual defaulters (only 3.44% of defaulters are correctly identified). This is a classic symptom of severe class imbalance — the model achieves high accuracy by simply predicting "no default" for most cases.

The class-weighted logistic regression improves recall to 67.93% but at the cost of precision (21.55%) and overall accuracy (67.56%).

## Key Statistical Findings

Based on the statistical analysis:

**Significant associations with default (Chi-square tests):**
- Employment Type had a significant association with default
- Having a Mortgage had a significant association with default
- Loan Purpose had a significant association with default

**Not statistically significant:**
- Marital Status was not statistically significant
- Education was not statistically significant
- Having Dependents was not statistically significant

**Mann-Whitney tests** reported significant distributional differences for:
- Age
- Income
- Loan Amount
- Credit Score
- Interest Rate

## Feature / Risk Insights

Based on the analysis (associations, not causal relationships):

- **Higher age** was associated with lower default risk
- **Higher income** was associated with lower default risk
- **Higher loan amount** was associated with higher default risk
- **Higher interest rate** was associated with higher default risk
- **Higher DTI ratio** was associated with higher default risk
- **Loan term** was reported as statistically insignificant in the logistic regression model

## Model Limitations

- **Class imbalance**: The dataset has a ~11.6% default rate, causing models to be biased toward the majority class
- **Very low recall**: The baseline model identifies only 3.44% of actual defaulters, resulting in a large number of false negatives
- **Logistic regression limitations**: Linear models may not capture nonlinear relationships and interactions between features
- **Dataset-specific generalizability**: Results are specific to this dataset and may not generalize to other lending contexts
- **Potential overfitting**: Complex models (Random Forest, XGBoost) may overfit to the training data
- **Interpretability vs model complexity trade-off**: More complex models may perform better but are harder to interpret

## Future Scope

The following are identified as future work (not yet implemented):

- **Random Forest** with hyperparameter tuning (currently only class-weighted)
- **Gradient Boosting** with more sophisticated hyperparameter optimization
- **Neural Networks** for capturing complex nonlinear patterns
- **Ensemble methods** combining multiple models
- **Real-time risk assessment** system integration
- **Fairness and bias analysis** across demographic groups
- **Calibration analysis** to improve probability estimates

## Repository Structure

```
loan-default-risk-analysis/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── README.md              # Data documentation
│   ├── raw/                   # Original datasets
│   │   ├── german_credit_raw.data
│   │   └── german_credit_clean.csv
│   └── processed/             # Cleaned/transformed datasets
│       ├── loan_default_clean.csv
│       ├── train_data.csv
│       └── test_data.csv
│
├── src/                       # Python source code
│   ├── main.py                # Combined pipeline (all phases)
│   ├── phase1_data_foundation.py
│   ├── phase2_eda.py
│   ├── phase3_statistical_tests.py
│   ├── phase3b_german_credit_validation.py
│   ├── phase4_data_prep.py
│   ├── phase5_model_comparison.py
│   ├── phase6_threshold_tuning.py
│   └── phase7_shap_interpretability.py
│
├── results/                   # Generated result CSVs
│   ├── phase5_model_comparison.csv
│   ├── phase6_threshold_tuning.csv
│   └── phase7_shap_importance.csv
│
├── figures/                   # Generated visualizations
│   ├── phase2_univariate_histograms.png
│   ├── phase2_boxplots_vs_default.png
│   ├── phase2_correlation_heatmap.png
│   ├── phase2_default_rate_by_category.png
│   ├── phase6_roc_curves.png
│   ├── phase6_pr_curves.png
│   ├── phase7_shap_bar.png
│   └── phase7_shap_summary.png
│
└── report/                    # Project reports
    ├── Loan_Default_Report.pdf
    └── Loan_Default_Revised_Report.docx
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tejasonawane01-dev/loan-default-risk-analysis.git
   cd loan-default-risk-analysis
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Running Individual Phases

Each phase script should be run from the `src/` directory:

```bash
cd src

# Phase 1: Data Foundation
python phase1_data_foundation.py

# Phase 2: Exploratory Data Analysis
python phase2_eda.py

# Phase 3: Statistical Hypothesis Testing
python phase3_statistical_tests.py

# Phase 3B: German Credit Validation
python phase3b_german_credit_validation.py

# Phase 4: Data Preparation
python phase4_data_prep.py

# Phase 5: Model Comparison
python phase5_model_comparison.py

# Phase 6: Threshold Tuning
python phase6_threshold_tuning.py

# Phase 7: SHAP Interpretability
python phase7_shap_interpretability.py
```

### Running the Combined Pipeline

```bash
cd src
python main.py
```

### Expected Directory Structure

The scripts expect the following directory structure:

```
loan-default-risk-analysis/
├── data/
│   ├── raw/
│   │   ├── german_credit_raw.data
│   │   └── german_credit_clean.csv
│   └── processed/
│       ├── loan_default_clean.csv
│       ├── train_data.csv
│       └── test_data.csv
├── src/
│   └── (Python scripts)
├── results/
│   └── (generated CSV files)
└── figures/
    └── (generated PNG files)
```

### Output Files

Running the scripts will generate:

- **Results**: `results/phase5_model_comparison.csv`, `results/phase6_threshold_tuning.csv`, `results/phase7_shap_importance.csv`
- **Figures**: `figures/phase2_*.png`, `figures/phase6_*.png`, `figures/phase7_*.png`

## Disclaimer

> **This is an academic/project analysis and should not be treated as a production credit-scoring system or used for real lending decisions without proper validation, governance, fairness assessment, and domain review.**

The models and findings presented in this project are for educational and research purposes only. They have not been validated for production use and should not be used to make actual lending decisions. Any real-world application of these techniques would require:

- Regulatory compliance review
- Fairness and bias assessment
- Model validation by qualified risk professionals
- Ongoing monitoring and recalibration
- Domain expert review

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **UCI Machine Learning Repository** for the Statlog German Credit Data (Hofmann, 1994)
- **Kaggle** for the Loan Default dataset
