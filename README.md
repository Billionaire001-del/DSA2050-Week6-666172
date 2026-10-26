# DSA2050 Week 6 — Regression and Baseline Comparison

**Student name:** Dancan Ratemo  
**Student ID:** 666172

## Lab objective
Estimate `Sales_KES` from advertising spend, store visits, discount, region, customer segment, and product category. Compare mean and median constant baselines with Linear Regression, Ridge Regression, and a shallow Decision Tree using MAE, RMSE, and R². The notebook also demonstrates leakage-safe preprocessing, training-only cross-validation, residual and subgroup error analysis, overfitting checks, and Ridge regularization tuning.

## Files
- `DSA2050_Week6_Guided_Regression_Lab.ipynb` — complete executable analysis, figures, metrics, and interpretations.
- `DSA2050_Week6_Sales_Data.csv` — the supplied lab dataset.

## Model comparison
The selected candidate is **Ridge Regression** (α = 1.0). It achieved test **MAE KES 13,714.40**, test RMSE KES 17,458.82, and R² 0.870. Its training-only five-fold CV MAE was KES 16,049.72 (fold standard deviation KES 1,315.68). The median baseline was the stronger constant baseline by test MAE at KES 39,146.75; Ridge reduced test MAE by KES 25,432.35 (65.0%) on this single held-out split.

## Two key findings
1. Ridge Regression ranked best on both the untouched test set and average training-only CV MAE, and materially outperformed the median baseline on the holdout set. This is promising for prediction, but the single split is not sufficient evidence for deployment.
2. Retail had the highest linear-model segment MAE (KES 15,560.58; n = 27). The unrestricted decision tree had training MAE KES 0.00 but test MAE KES 26,797.50, a clear sign of overfitting; its nearly perfect training fit did not generalize.

## Limitations
1. The dataset has 320 observations and may be a teaching/synthetic sample, so its results may not generalize to real business operations.
2. Important variables such as time/seasonality, store context, product mix or price, inventory, and competitor activity are absent.
3. A random holdout split does not test future-period data drift; prediction does not establish causation. Segment sample sizes are also modest.

## Reproduction
Install Python dependencies (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, and Jupyter), then run all notebook cells from top to bottom with the CSV in the same directory. The notebook fixes the random state at 42 and keeps the final 20% test split separate from training-only cross-validation.
