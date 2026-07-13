# Part 2 — Supervised Machine Learning: Build, Train, and Evaluate

Builds on `cleaned_data.csv` from Part 1 (Loan Approval / Credit Risk dataset, 614 rows).

## Label Definitions

- **Feature matrix `X`**: all columns except `Loan_ID` (identifier, not a feature), `LoanAmount`, and
  `Loan_Status` (both targets are excluded from `X` to prevent either model from leaking the other's answer).
- **Regression label `y_reg`**: `LoanAmount` (continuous, thousands).
- **Classification label `y_clf`**: `Loan_Status` mapped to 1 (`Y`, approved) / 0 (`N`, not approved) — a
  **natural binary column already in the dataset**, not derived by binarizing a continuous column at its
  median.

## Preprocessing Decisions

- **Encoding**:
  - `Education` → **label/ordinal encoding** (`Not Graduate=0`, `Graduate=1`). Treated as ordinal because
    higher educational attainment plausibly correlates with higher, more stable earning potential — directly
    relevant to loan underwriting — so the order carries real information.
  - `Gender`, `Married`, `Self_Employed`, `Property_Area` → **one-hot encoding** (`drop_first=True`). These
    have no natural ranking (e.g. Urban vs Semiurban vs Rural), so label-encoding them as 0/1/2 would falsely
    imply an order that a linear/logistic model would treat as meaningful. `Dependents` and `Credit_History`
    were already numeric from Part 1 and needed no further encoding.
  - Final feature set (11 columns): `Dependents, Education, ApplicantIncome, CoapplicantIncome,
    Loan_Amount_Term, Credit_History, Gender_Male, Married_Yes, Self_Employed_Yes, Property_Area_Semiurban,
    Property_Area_Urban`.
- **Split**: a single `train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)` call so the same 491
  training rows / 123 test rows are used for both models.
- **Scaling**: `StandardScaler` was **fit only on `X_train`**, then used to transform both `X_train` and
  `X_test`. Fitting the scaler on the full dataset (or on the test set) would be data leakage — the scaler's
  mean/std would encode test-set statistics into the transformation applied to training data, giving an
  optimistically biased estimate of test performance.

## Regression: Linear Regression vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 2632.35 | 0.5164 |
| Ridge (alpha=1.0) | 2633.13 | 0.5163 |

**Top 3 features by absolute coefficient** (Linear Regression, on scaled features): `ApplicantIncome` (+45.52),
`CoapplicantIncome` (+20.94), `Loan_Amount_Term` (+8.17). A large **positive** coefficient means a
one-standard-deviation increase in that (scaled) feature is associated with that many units' increase in
predicted `LoanAmount`, holding other features fixed — e.g. higher-income applicants are predicted to receive
larger loans. A large **negative** coefficient would mean the opposite.

**Ridge vs OLS**: the two models are nearly identical here (MSE/R² differ in the third decimal place). Ridge
adds an L2 penalty (`alpha × Σcoef²`) that shrinks coefficients toward zero to control variance from correlated
features; at `alpha=1.0` that penalty is mild relative to the scale of this problem, so it barely moves the
fit. `alpha` controls the strength of that shrinkage — `alpha=0` recovers OLS exactly, and larger values would
shrink the coefficients (and likely raise MSE slightly) more noticeably.

## Classification: Logistic Regression

**Class imbalance**: `Loan_Status` in the training set is 69.7% approved / 30.3% not-approved — below the 35%
threshold. We handled this with **`class_weight='balanced'`** rather than SMOTE: it re-weights the loss
function inversely to class frequency, avoids generating synthetic applicant records in a financial dataset,
and needs no extra dependency.

**Baseline (C=1.0) results:**

- Confusion matrix: `[[22, 21], [9, 71]]` (rows = actual 0/1, columns = predicted 0/1)
- Accuracy: **0.7561**
- Precision / Recall / F1 (class 1 = approved): **0.77 / 0.89 / 0.83**
- Precision / Recall / F1 (class 0 = not approved): **0.71 / 0.51 / 0.59**
- **AUC: 0.7331**

**Precision** = TP / (TP + FP) — of everyone predicted "approved," the fraction actually approved.
**Recall** = TP / (TP + FN) — of everyone actually approved, the fraction the model caught.

**Which matters more here**: a lender worried primarily about default risk would rather miss a few good
applicants than approve bad ones, which makes **precision** on the "approved" prediction the more important
metric in this domain — a false positive (wrongly predicting approval) risks a bad loan; a false negative
(wrongly predicting rejection) only risks losing a customer, not money.

**AUC interpretation**: an AUC of 0.733 means that if you picked one randomly approved and one randomly
rejected applicant from the test set, the model ranks the approved applicant's predicted probability higher
about 73% of the time — clearly better than the 0.5 random baseline, but leaving real room for the ensemble
models in Part 3 to improve on.

### Decision-Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.7596 | 0.9875 | **0.8587** |
| 0.40 | 0.7500 | 0.9375 | 0.8333 |
| 0.50 | 0.7717 | 0.8875 | 0.8256 |
| 0.60 | 0.7639 | 0.6875 | 0.7237 |
| 0.70 | 0.8158 | 0.3875 | 0.5254 |

The threshold that **maximizes F1** on this test set is **0.30** (F1 = 0.8587) — lower than the default 0.5,
because `class_weight='balanced'` already shifts the model's predicted probabilities, so a slightly lower cut
recovers more true positives without sacrificing much precision. However, given the domain reasoning above
(precision matters more for a default-risk-averse lender), we would actually consider **raising** the threshold
above 0.5 in practice rather than chasing the F1-maximizing point — the cost of doing so is a lower recall
(more creditworthy applicants incorrectly rejected), a trade-off a real lender would weigh against their risk
appetite rather than optimizing blindly for F1.

### Regularization Experiment: C=0.01 vs C=1.0

| Model | Precision | Recall | AUC |
|---|---|---|---|
| Baseline (C=1.0) | 0.7717 | 0.8875 | 0.7331 |
| Strong regularization (C=0.01) | 0.7708 | 0.9250 | 0.7375 |

`C` is the **inverse** of the regularization strength in `sklearn`'s logistic regression — a smaller `C` means
a stronger L2 penalty, shrinking coefficients toward zero and simplifying the decision boundary. Here, the
strongly regularized model (`C=0.01`) actually performs marginally **better** on recall and AUC, suggesting the
baseline model was carrying a small amount of unnecessary variance/overfitting that the extra shrinkage helps
control — though the difference is small enough that its reliability needed to be checked statistically (next
section).

### Bootstrap Confidence Interval for the AUC Difference

- 500 bootstrap resamples of the test set (with replacement), AUC difference = AUC(C=1.0) − AUC(C=0.01)
  computed per resample.
- **Mean AUC difference: −0.0043**
- **95% CI: [−0.0263, 0.0218]**
- **The interval includes zero.**

**Interpretation**: because the confidence interval spans zero, the small AUC difference observed between the
two regularization strengths is **not statistically reliable** at the 95% level — on this test set, we cannot
confidently say either model outperforms the other. This is a useful finding in itself: it means the
regularization strength isn't the lever that will meaningfully move this model's performance, and more signal
is likely to come from better features or a different model family (explored in Part 3) than from further
tuning `C`.

## Repository Contents

- `part2_loan_models.ipynb` — full notebook, executed end-to-end, all outputs and the ROC curve visible without
  re-running it.
- `cleaned_data.csv` — input from Part 1.
- `README.md` — this file.
