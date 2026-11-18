# AI Usage & Its Impact on Students and Professionals

An end-to-end, leakage-safe data-science project that explains and predicts **self-reported productivity gain from generative-AI tools**, and identifies which users are likely to **recommend AI** to others.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-End--to--End-brightgreen)

---

##  Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Key Insights & Recommendations](#-key-insights--recommendations)
- [Is the Model Trustworthy?](#-is-the-model-trustworthy)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Author](#-author)

---

## Problem Statement

Generative-AI tools (ChatGPT, Claude, Gemini, Copilot and others) are now part of everyday study and work, yet the benefit people report varies enormously. Educators, employers and tool vendors need evidence on:

1. **What drives self-reported productivity gain?** Usage volume, tool choice, purpose, user profile, or AI performance?
2. **Can we predict `Productivity_Score`** so low-benefit users can be supported early?
3. **Who will recommend AI to others** (a proxy for sustained adoption)?

| Task | Target | Type |
|---|---|---|
| **Primary** | `Productivity_Score` (1 to 10) | Regression |
| **Secondary** | `Would_Recommend == "Yes"` | Binary classification |

---

## Dataset

- **File:** `AI_Usage_and_Impact_on_Students_and_Professionals.csv`
- **Size:** 300 survey rows × 19 columns → **274 usable rows** after cleaning

**Why it is hard**
- Only a few hundred rows, so over-fitting is easy and honest validation is critical
- Very messy data: units embedded in strings (`"2 hrs"`, `"$20"`, `"30 yrs"`), inconsistent labels (`ChatGPT / Chat GPT / chatgpt`, `U.K. / United Kingdom`), multilingual job titles, duplicate IDs, impossible values and missing values in every column
- **Leakage traps:** satisfaction, time saved and perceived accuracy are measured at the same time as the target

**Cleaning highlights**
- Parsed units and symbols; harmonised gender, country, education and tool labels
- Mapped **126 free-text professions to 7 role families**
- Blanked 6 impossible values (age 97 ×2, income 250,000, 23 AI-hrs/day, 20 work-hrs ×2)
- Removed 12 duplicate respondents and 14 rows with no target (the target is never imputed)

---

## Approach

1. **Audit & clean** with every change logged
2. **EDA & statistical tests:** Mann-Whitney, Kruskal-Wallis, Spearman, with Benjamini-Hochberg FDR correction
3. **Feature engineering with explicit maths:** log(1+x) transforms, squares for diminishing returns, AI-share of working time, cost-to-income ratio, free-text statistics, and TF-IDF → truncated-SVD topics (84 encoded inputs)
4. **Split first, fit later:** imputers, scalers, encoders, TF-IDF, PCA and selectors all live inside scikit-learn `Pipeline`s and are refitted on each training fold only
5. **Scaling comparison:** none / standard / robust / Yeo-Johnson (Yeo-Johnson won on CV error for linear, KNN and SVR models)
6. **Feature reduction:** variance filter, correlation, VIF, PCA (verified against a manual eigen-decomposition), Lasso path
7. **Modelling:** 11 regressors, repeated 5×3 CV, randomised/grid tuning, **nested CV** for an unbiased estimate of the tuned model
8. **Diagnostics & leakage audit:** learning and validation curves, target-shuffle test, train/test overlap check, with/without-outcome-features experiment, bootstrap CIs
9. **Interpretation** with permutation importance and linear-model coefficients

---

## Key Results

### Primary task: Productivity regression (55-row untouched test set)

| Item | Result |
|---|---|
| **Selected model** | Tuned **Random Forest** (leak-free feature set) |
| **Nested-CV R²** | ≈ 0.60 (≈ 37% lower RMSE than the baseline) |
| **Test R²** | ≈ 0.60 (95% bootstrap CI ≈ 0.42 to 0.74) |
| **Test MAE** | ≈ 1.0 point on a 10-point scale |
| **Permutation test** | p ≈ 0.01 (shuffled-target R² ≈ −0.08) |

### Secondary task: Will the user recommend AI?

| Item | Result |
|---|---|
| **Model** | Shallow Gradient Boosting |
| **Test ROC-AUC** | ≈ 0.82 (95% CI ≈ 0.69 to 0.94) |
| **Dominant predictor** | Satisfaction (shuffling it drops test AUC by ≈ 0.25) |

---

## Key Insights & Recommendations

1. **Drive engagement, not tool choice.** Daily AI hours is by far the strongest correlate of productivity (ρ ≈ 0.81). Tool, purpose, profession family, education, country, gender, age, income and student/professional status show **no significant differences** after FDR correction.
2. **Premium tiers don't buy productivity.** Monthly AI cost is unrelated to productivity (ρ ≈ −0.07).
3. **Don't use "perceived accuracy" as a success KPI** (ρ ≈ 0.07). Track usage time, time saved and satisfaction instead.
4. **Use the model as a coarse screening tool.** An average error of about 1 point suits flagging low-benefit users for coaching, not ranking individuals.
5. **To grow advocacy, improve satisfaction.** UX, support and reliability work should pay off in word-of-mouth adoption.
6. **Be careful with heavy use.** Very high daily hours are rare, so conclusions there are uncertain, and the survey has no wellbeing or output-quality measures.
7. **Improve data collection:** dropdowns, validated numeric fields, unique respondent IDs, standard lists, and objective measures such as task completion time.
8. **Establish causality before big decisions.** A pilot with before/after or randomised comparison would settle whether heavy users are more productive *because of* AI.

---

## Is the Model Trustworthy?

- **Not under-fitted:** ~37% lower RMSE than the mean baseline, and the learning curve flattens.
- **Not materially over-fitted:** the tuned model's train-to-CV R² gap is ≈ 0.12, and CV (≈ 0.59) and test (≈ 0.60) R² agree.
- **No leakage found:** the split precedes every fitted step, all fitted steps sit inside pipelines, there is no train/test overlap, and the target-shuffle test collapses to ≈ 0.
- **Outcome columns excluded on purpose.** Adding accuracy, satisfaction and time saved lifts test R² only from ≈ 0.60 to ≈ 0.63, and they would not be available at prediction time.
- **Caveat:** the 55-row test set is more spread out than the training data, so absolute test RMSE (≈ 1.3) looks worse than CV RMSE (≈ 1.1) even though R² matches. The confidence interval is wide for the same reason.

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<my-username>/<my-repo-name>.git
cd <my-repo-name>
```

### 2. Install dependencies
```
pip install pandas numpy scipy scikit-learn matplotlib seaborn if not using jupyter notebook
```

### 3. Add the data
Place `AI_Usage_and_Impact_on_Students_and_Professionals.csv` next to the notebook (or edit `DATA_PATH` in the setup cell).

### 4. Run the notebook
```bash
jupyter notebook AI_Usage_Impact_Analysis.ipynb
```
Choose **Kernel → Restart & Run All**.

---

## Repository Structure

```
├── AI_Usage_Impact_Analysis.ipynb                          # Full end-to-end notebook
├── AI_Usage_and_Impact_on_Students_and_Professionals.csv   # Survey data (add locally)
└── README.md
```

---

##  Limitations & Next Steps

**Limitations**
- Small sample (274 rows), so uncertainty is wide
- Self-reported outcomes are subject to response bias and strongly correlated with each other
- Cross-sectional survey, so no causal claims

**Next steps**
- Collect more data and report intervals
- Monotonic-constrained boosting for the usage-hours effect
- SHAP explanations
- Calibrated classifier probabilities
- Drift monitoring if deployed

---

##  Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
