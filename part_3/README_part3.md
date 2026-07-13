# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

Builds on the classification task from Part 2 (`Loan_Status`), using the identical encoding/split/scaling
reproduced at the top of the notebook so `X_train_scaled`, `X_test_scaled`, `y_clf_train`, `y_clf_test` match
Part 2 exactly.

## Decision Tree: Unconstrained vs Controlled

| Tree | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| Unconstrained (`max_depth=None`) | 1.0000 | 0.7073 | **0.2927** |
| Controlled (`max_depth=5`, `min_samples_split=20`) | 0.8289 | 0.7642 | **0.0647** |

The unconstrained tree memorizes the training set perfectly (100% train accuracy) and drops nearly 30 points on
the test set — textbook **overfitting**. Decision trees are high-variance models because each split is chosen
greedily for the data at that node, with no mechanism to revisit or correct earlier splits; left unconstrained,
the tree keeps partitioning until it's fit noise specific to the training rows. `max_depth` limits how many
splits deep any path can go (trading some bias for much lower variance), and `min_samples_split=20` stops the
tree from creating splits that respond to fewer than 20 samples — both of which shrink the train-test gap from
0.29 to 0.06.

## Gini vs Entropy (max_depth=5)

| Criterion | Test Accuracy |
|---|---|
| Gini (`1 - Σpᵢ²`) | 0.7398 |
| Entropy (`-Σpᵢ log₂(pᵢ)`) | **0.7642** |

Both formulas measure how mixed a node's classes are; a node with **Gini = 0** contains samples from only one
class (perfectly pure). Entropy edged out Gini slightly here, though in general the two criteria tend to
produce very similar trees — Gini is just cheaper to compute since it avoids the logarithm.

## Random Forest

- Train accuracy: **0.9246** | Test accuracy: **0.7805** | Test AUC: **0.7501**

**Top 5 features by importance**: `Credit_History` (0.340), `ApplicantIncome` (0.235), `CoapplicantIncome`
(0.132), `Loan_Amount_Term` (0.072), `Dependents` (0.058). `Credit_History` dominating the importance ranking
lines up with real-world lending intuition — it's the single strongest predictor of approval.

Random Forest computes feature importance by averaging each feature's Gini-impurity reduction across every
split that used it, across all trees — different from a linear regression coefficient, which measures a
feature's *linear, additive* association with the target holding others fixed. Importance instead reflects how
much a feature *actually reduced impurity in practice*, across many non-linear splits.

**Bagging**: each tree trains on a bootstrap sample (drawn with replacement) and, at each split, considers only
a random subset of √(n_features) candidate features. This decorrelates the individual trees, so averaging their
predictions cancels out much of each tree's overfitting — the ensemble's train-test gap (0.9246 → 0.7805) is
already smaller than the unconstrained single tree's, without needing an explicit `max_depth` cap.

## Gradient Boosting

- Train accuracy: **0.8839** | Test accuracy: **0.7398** | Test AUC: **0.6663**

Notably, Gradient Boosting underperformed both Random Forest and even plain Logistic Regression on this
dataset's test AUC — a reminder that "more sophisticated" doesn't automatically mean "better" on a small
dataset (491 training rows): boosting's sequential error-correction can overfit faster than bagging on limited
data, and its default hyperparameters here weren't tuned.

## Feature Ablation Study

5 lowest-importance features removed: `Married_Yes`, `Education`, `Self_Employed_Yes`, `Gender_Male`,
`Property_Area_Urban`.

| Model | Test AUC |
|---|---|
| Full-feature Random Forest | 0.7501 |
| Reduced-feature Random Forest | **0.7557** |
| Difference | +0.0055 |

Removing these 5 features **did not hurt** performance — AUC held steady (marginally improved), confirming
they were genuinely close to noise for this model rather than contributing real signal. **Production
implication**: dropping them gives a simpler, 6-feature model with lower inference cost and one less data
pipeline to maintain per removed field, at essentially zero accuracy cost here — a reasonable trade to make
before deployment.

## Cross-Validated Comparison (5-fold, ROC-AUC, on training data)

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.7586 | 0.0345 |
| Decision Tree (depth=5) | 0.7229 | 0.0486 |
| **Random Forest** | **0.7717** | 0.0561 |
| Gradient Boosting | 0.7474 | 0.0446 |

Random Forest has the best CV mean AUC, though also the highest variance across folds. A single train-test
split gives one estimate that depends heavily on which rows land in a small 123-row test set; 5-fold CV instead
trains and scores the model 5 times over different partitions, giving both a more stable mean estimate and a
sense of how much that estimate varies fold-to-fold — the same kind of reliability check the bootstrap analysis
in Part 2 was doing for the regularization comparison.

## GridSearchCV Tuning

- **Best params**: `n_estimators=100`, `max_depth=10`, `min_samples_leaf=1`
- **Best CV score (ROC-AUC)**: 0.7714
- **Total configurations evaluated**: 3 × 3 × 2 = 18 param combinations × 5 folds = **90 model fits**

Grid Search exhaustively checks every combination in the grid, guaranteeing the best combination *within that
grid* is found — but the fit count grows multiplicatively with every added hyperparameter or value, so it gets
expensive fast for larger grids. Randomized Search instead samples a fixed number of random combinations,
scaling much better to larger search spaces at the cost of no longer guaranteeing the single best combination
is checked. Here, the grid was small enough (18 combos) that exhaustive search was cheap and appropriate.

## Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 1.0000 | 0.7701 |
| 0.4 | 1.0000 | 0.7788 |
| 0.6 | 0.9994 | 0.7526 |
| 0.8 | 0.9977 | 0.7881 |
| 1.0 | 0.9956 | 0.7501 |

(i) Training AUC starts near-perfect and drifts down only slightly as more rows are added — expected for a
high-variance model like Random Forest, which can nearly memorize very small training sets.
(ii) Test AUC does **not** show a clean upward trend with more data — it fluctuates between ~0.75 and ~0.79
without a clear rising pattern, which is at least partly noise from the small, fixed 123-row test set.
(iii) **Conclusion**: the model looks **capacity/feature-limited rather than data-limited** — test AUC has
plateaued in the 0.75–0.79 range rather than climbing steadily toward 1.0 as training data grows, so further
gains are more likely to come from richer features (e.g. debt-to-income ratio, employment length) than from
simply collecting more rows of the current 11 features.

## Serialized Model

`best_model.pkl` — the tuned Random Forest pipeline (`SimpleImputer` → `StandardScaler` → `RandomForestClassifier`)
from GridSearchCV, saved with `joblib.dump`. Reload-and-predict was verified on two hand-crafted applicant
profiles:

- **Profile 1** (income 6000, co-applicant income 2000, credit history = 1, semiurban): predicted **Approved**,
  probability 0.753.
- **Profile 2** (income 1800, no co-applicant income, credit history = 0, urban, self-employed): predicted
  **Not Approved**, probability 0.110.

Both predictions align with the feature-importance findings above — `Credit_History` and `ApplicantIncome`
dominate the decision, and the two profiles differ sharply on exactly those two features.

## Summary Comparison Table

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.7586 | 0.0345 | 0.7331 |
| Decision Tree (depth=5) | 0.7229 | 0.0486 | 0.6901 |
| Random Forest | 0.7717 | 0.0561 | 0.7501 |
| Gradient Boosting | 0.7474 | 0.0446 | 0.6663 |
| **Tuned Random Forest (GridSearchCV)** | **0.7714** | 0.0563 | **0.7501** |

## Recommendation

**Random Forest (tuned via GridSearchCV)** is the recommended model. It has the highest test-set AUC (tied
with the untuned Random Forest, since GridSearchCV's best parameters landed close to the original manual
settings), the highest CV mean AUC, and — unlike the standalone models — it's packaged as a single reusable
`sklearn` Pipeline with imputation and scaling baked in, so it can be deployed and reloaded exactly as scored
here without re-implementing preprocessing separately. Gradient Boosting is not recommended for this dataset
in its current, untuned form — it underperformed even Logistic Regression on test AUC, likely overfitting the
small training set faster than the bagged Random Forest did.

## Repository Contents

- `part3_loan_ensembles.ipynb` — full notebook, executed end-to-end, all outputs visible without re-running it.
- `best_model.pkl` — serialized tuned Random Forest pipeline.
- `cleaned_data.csv` — input from Part 1.
- `README.md` — this file.
