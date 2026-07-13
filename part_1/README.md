# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset Description

**Loan Approval / Credit Risk dataset** — 614 loan applications, 13 columns. Each row is one applicant;
the dataset records demographic and financial fields (income, co-applicant income, education, employment,
dependents, credit history, property area) plus the loan outcome fields (`LoanAmount`, `Loan_Amount_Term`,
`Loan_Status`).

- **Source**: [`shrikant-temburwar/Loan-Prediction-Dataset`](https://github.com/shrikant-temburwar/Loan-Prediction-Dataset)
  (`train.csv`), a public mirror of the Analytics Vidhya "Loan Prediction Practice Problem" dataset. The raw file
  is included in this repository as `loan_data.csv`.
- **Why this dataset**: it gives every downstream part of the capstone something real to work with —
  `Loan_Status` is a ready-made binary classification target, `LoanAmount` is a ready-made regression target,
  there's a genuine mix of ordinal (`Dependents`, `Credit_History`) and nominal (`Property_Area`, `Gender`)
  categorical columns, real missing values spread across seven columns, and right-skewed income columns with
  real outliers — none of it synthetic or manufactured for the assignment.

Columns:

| Column | Type | Description |
|---|---|---|
| Loan_ID | identifier | unique application ID |
| Gender | categorical (nominal) | Male / Female |
| Married | categorical (nominal) | Yes / No |
| Dependents | categorical (ordinal) | 0, 1, 2, 3+ |
| Education | categorical (nominal) | Graduate / Not Graduate |
| Self_Employed | categorical (nominal) | Yes / No |
| ApplicantIncome | numeric | applicant's monthly income |
| CoapplicantIncome | numeric | co-applicant's monthly income |
| LoanAmount | numeric | loan amount requested (thousands) |
| Loan_Amount_Term | numeric | loan term, in months |
| Credit_History | numeric (binary) | 1 = meets credit guidelines, 0 = does not |
| Property_Area | categorical (nominal) | Urban / Semiurban / Rural |
| Loan_Status | categorical (binary target) | Y / N — loan approved |

## Steps Taken

1. **Loaded** the raw CSV with `pd.read_csv`, inspected shape (614 × 13), `.head()`, and `.dtypes`.
2. **Null analysis**: computed count and percentage of missing values per column. **No column exceeded the
   20% null threshold** — the worst offender, `Credit_History`, is at 8.14%. Numeric columns below the
   threshold (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) were filled with the column **median**.
   Categorical columns with nulls (`Gender`, `Married`, `Dependents`, `Self_Employed`) were filled with the
   column **mode** for completeness, so no nulls carry forward into dtype correction or Part 2 encoding.
   - **Why median, not mean**: `LoanAmount` (and the income columns) are right-skewed — a small number of
     large loans/incomes pull the mean upward, so it no longer represents a typical applicant. The median is
     robust to that skew.
3. **Duplicates**: `df.duplicated().sum()` found **0 duplicate rows** (each row has a unique `Loan_ID`), so no
   rows were removed and the null percentages from step 2 are unaffected.
4. **Data type correction**: `Dependents` was stored as `object` because one category is the string `"3+"`.
   Converted it to numeric (`"3+"` → `3`, then `pd.to_numeric`). Converted six repetitive string columns
   (`Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`) to `category` dtype.
   **Memory usage dropped from ~285 KB to ~69 KB (a ~76% reduction)**, because `category` dtype stores each
   unique string once and references it by an integer code instead of repeating full strings per row.
5. **Descriptive statistics & skewness**: ran `.describe()` on all numeric columns and computed `.skew()` for
   each. **`CoapplicantIncome` has the highest absolute skewness (≈7.49)**, followed by `ApplicantIncome`
   (≈6.54) — both **positively skewed**: most applicants cluster at low-to-moderate income, with a long right
   tail of a few very high earners. This matters for imputation because the **mean** gets pulled upward by
   those extreme values and misrepresents the typical applicant, while the **median** stays representative —
   which is why median imputation was used throughout.
6. **Outlier detection (IQR)** on `ApplicantIncome` and `LoanAmount`:

   | Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers |
   |---|---|---|---|---|---|---|
   | ApplicantIncome | 2,877.5 | 5,795.0 | 2,917.5 | −1,498.75 | 10,171.25 | **50 rows** |
   | LoanAmount | 100.0 | 168.0 | 68.0 | −2.0 | 270.0 | **39 rows** |

   Outliers were **not dropped** — high-income applicants and large loans are a real population segment, not
   data errors. **Plan for Part 2**: cap (winsorize) these values at the IQR bounds before feeding them into
   the linear/logistic regression models, which are sensitive to extreme values; retain them uncapped for the
   tree-based models in Part 3, which split on thresholds and are naturally robust to outliers.
7. **Visualizations** (all six required plots, all reproduced in the notebook with real rendered output):
   - **Line plot** — sorted `ApplicantIncome` values, showing the steep right-tail rise typical of skewed
     income data.
   - **Bar chart** — mean `LoanAmount` by `Property_Area`, showing Rural applicants have the highest mean
     loan amount.
   - **Histogram** — `CoapplicantIncome` (the most-skewed column): a tall spike near zero (many applicants
     have no co-applicant income) with a long, thin right tail — consistent with the skew score of ≈7.49.
   - **Scatter plot** — `ApplicantIncome` vs `LoanAmount`: a moderate positive relationship (Pearson r ≈ 0.57).
     Higher-income applicants tend to get larger loans, though the relationship has meaningful scatter and is
     not tight.
   - **Box plot** — `LoanAmount` by `Education`: Graduates have a slightly higher median loan amount and a
     visibly wider spread with more high-value outliers than Non-Graduates.
   - **Correlation heat map** — of all five numeric columns. The strongest pair is `ApplicantIncome` vs
     `LoanAmount` (r ≈ 0.57). This is unlikely to be purely coincidental, but it isn't necessarily direct
     causation either: a plausible alternative explanation is that both are jointly driven by the bank's
     underwriting process, which sizes the loan to what the applicant's overall financial profile (income,
     assets, existing debt — not fully captured in this dataset) can plausibly support.
8. **Extended analysis**:
   - **(a) Mean vs median comparison** for the two most-skewed columns (`CoapplicantIncome`, `ApplicantIncome`):
     for `ApplicantIncome`, mean ≈ 5,403 vs median ≈ 3,812 — a large gap confirming the right-skew. **Median**
     was chosen for imputation for the same skew-robustness reason given in step 2. Both columns had zero
     missing values originally, so `fillna()` was a no-op in practice; `isnull().sum()` confirms zero nulls
     remain, and the logic would apply automatically to a future data extract with real gaps in these columns.
   - **(b) Spearman vs Pearson**: the three pairs with the largest |Spearman − Pearson| difference were
     `ApplicantIncome` vs `CoapplicantIncome` (diff ≈ 0.20, |Spearman| > |Pearson| — a consistent but
     non-proportional monotonic relationship, likely driven by shared outlier applicants), `ApplicantIncome`
     vs `Credit_History` (diff ≈ 0.06), and `ApplicantIncome` vs `LoanAmount` (diff ≈ 0.06, |Pearson| ≥
     |Spearman| — closer to linear, matching the scatter plot). **Pearson** will guide feature selection for
     the linear/logistic models in Part 2; Spearman is kept as a secondary check for the tree-based models in
     Part 3.
   - **(c) Grouped aggregation** — `LoanAmount` grouped by `Property_Area`: Rural has the highest mean
     (≈152.3), Urban has the highest standard deviation (≈94.5, meaning Urban loan amounts are the most
     spread out). A high within-group std **is** a modeling concern — knowing only "this applicant is Urban"
     gives little precision about their specific loan amount on its own. The ratio of highest to lowest group
     mean is **≈1.07** (only ~7% difference), suggesting `Property_Area` alone carries **weak** predictive
     signal and would need to be combined with other features.
9. Saved the fully cleaned, de-duplicated, correctly-typed, zero-null dataset to `cleaned_data.csv` for use in
   Parts 2 and 3.

## Repository Contents

- `part1_loan_eda.ipynb` — full notebook , need to run all to get cleaned data and plots visualization
- `loan_data.csv` — the raw input dataset.
- `cleaned_data.csv` — the cleaned output dataset (appears when ececuted).
- `README.md` — this file.
