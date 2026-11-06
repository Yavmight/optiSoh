# Pre-Declared Experiment & Statistical Protocol

## 1. Confirmatory Primary Hypothesis
**H1**: The Pareto knee-selected solution from the joint co-optimizer (NSGA-II / Hybrid) achieves a lower mean cross-cell RMSE than the best-performing two-stage baseline (LASSO, MI, or SHAP-RFE + tuned stack) under outer Leave-One-Cell-Out (LOCO) evaluation.
- **Statistical Test**: One-sided paired Wilcoxon signed-rank test pooled over all cells across NASA, Oxford, and Sandia.
- **Significance Threshold**: $\alpha = 0.05$.

## 2. Secondary Exploratory Analyses
- Pairwise Wilcoxon tests with Holm step-down multiplicity correction across all methods.
- Friedman omnibus test and Demšar critical-difference diagram (Nemenyi post-hoc).
- Cross-dataset zero-shot transfer RMSE and Conformal Prediction interval coverage (target 90% and 95%).
- Negative result logging: Full reporting of configurations where joint optimization does not beat default ensembles.
