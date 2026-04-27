# Predicting Diabetes from Health and Lifestyle Indicators

Support Vector Machine classifiers (linear, RBF, and polynomial kernels) trained to predict diabetes status using demographic and lifestyle data from the 2022 National Health Interview Survey.

## Project Overview

Diabetes affects roughly 10% of US adults and is strongly associated with modifiable lifestyle factors. This project builds three SVM classifiers using only readily-available demographic and self-reported lifestyle variables, evaluating whether nonlinear decision boundaries improve on a linear model and identifying which factors carry the most predictive signal.

## Data

- **Source:** [IPUMS NHIS 2022 extract](https://nhis.ipums.org/nhis/)
- **Sample size:** 24,109 adults aged 18-85 after cleaning
- **Target variable:** `DIABETICEV` (ever diagnosed with diabetes)
- **Class prevalence:** 10.4% positive

## Predictors

| Variable | Description |
|---|---|
| `BMICALC` | Body Mass Index |
| `AGE` | Age in years (topcoded at 85) |
| `MOD10DMIN` | Minutes per session of moderate physical activity |
| `VIG10DMIN` | Minutes per session of vigorous physical activity |
| `HRSLEEP` | Usual hours of sleep per day |
| `SODAPNO` | Soda consumption frequency |
| `POVERTY_RATIO` | Income relative to federal poverty threshold |
| `MALE` | Binary sex indicator |

## Methodology

- 80/20 stratified train/test split with feature standardization fit on training data only
- 3-fold cross-validation over hyperparameter grids for each kernel
- ROC-AUC as the tuning metric (chosen over accuracy due to class imbalance)
- `class_weight='balanced'` applied to handle the 10:1 minority class ratio

## Results

| Kernel | Best Hyperparameters | Test AUC |
|---|---|---|
| Linear | C=1 | 0.7818 |
| RBF | C=1, γ=0.01 | 0.7894 |
| Polynomial | C=1, degree=3, γ=0.01 | 0.7898 |

All three kernels achieve comparable performance (within 0.01 AUC of each other), suggesting the underlying relationship is well-captured by smooth, near-linear decision surfaces. AGE and BMI emerge as the dominant predictors. Physical activity provides additional signal. Sleep duration and sex show no marginal predictive value at the population level.

## Key Findings

- Diabetes risk in this dataset is dominated by a joint effect of age and BMI, with each acting as a sufficient individual risk factor at the extremes
- Soda consumption shows an unexpected negative association with diabetes status, likely reflecting reverse causation (post-diagnosis dietary changes), illustrating a limitation of cross-sectional data for causal inference
- The convergence of three structurally different kernels to nearly identical performance suggests linear models are an appropriate choice for this prediction problem when interpretability is valued

## Files

- `diabetes_svm_analysis.ipynb` - Main analysis notebook (data cleaning, model fitting, evaluation, and visualization)
- `figures/` - Generated plots (predictor distributions, ROC curves, decision boundaries)

## Reproducibility

All random seeds are fixed at 42 throughout the notebook. To reproduce, download the NHIS 2022 extract from the IPUMS link above (free account required) and update the file path in the data load cell.

## Tools

Python, scikit-learn, pandas, matplotlib
