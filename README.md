# sohML: Partial-Charging Battery SOH Co-Optimization

A reproducible, leakage-safe machine learning and optimization framework for **Window-Adaptive, Multi-Objective Metaheuristic Co-Optimization of Features and Hyperparameters for Partial-Charging Battery State-of-Health (SOH) Regression**.

---

## 🔬 Core Research Question
Does jointly optimizing the **partial-charging voltage window $[V_{\text{start}}, V_{\text{end}}]$**, a **feature mask**, and **regressor hyperparameters** via Pareto multi-objective search beat equally tuned simple baselines under:
1. **Leave-One-Cell-Out (LOCO) cross-validation** on unseen battery cells?
2. **Cross-dataset transfer** across chemistries and operating conditions?

---

## 👥 Team Work Division
- **Data Engineer 1 (DE1)**: Raw data ingestion, constant-current QC, physics normalization, 2 mV voltage grid, dataset coverage audits, cross-dataset transfer harness (`src/sohml/data/`, `evaluation/transfer.py`).
- **Data Engineer 2 (DE2)**: 40+ candidate health indicators (`dQ/dV`, `dT/dV`, Sample Entropy), quantized window cache, leakage-safe splitters, two-stage baselines (LASSO, MI, SHAP-RFE), statistical evaluation suite (`src/sohml/features/`, `splits/`, `baselines/`, `evaluation/statistics.py`).
- **Model Trainer (ML1)**: Stacked regressor (`SVR + RF + XGBoost -> Ridge`), multi-objective optimizers (NSGA-II, NSGA-III, Hybrid DE-PSO/HHO-GWO), Pareto knee selection, joint baselines (Random, Optuna TPE), run orchestration & conformal prediction intervals (`src/sohml/models/`, `optimization/`, `orchestration/`, `evaluation/conformal.py`).

See `TEAM_WORK_DIVISION.md` for daily contracts, interface boundaries, and milestone gates.

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -e .
```

### 2. Run Smoke Test (Synthetic Data)
```bash
python scripts/run_experiment.py --mode smoke --budget 24 --seeds 2
```

### 3. Run Unit & Zero-Leakage Tests
```bash
pytest tests -v
```

### 4. Benchmark Evaluation Cost on Local Hardware
```bash
python scripts/benchmark_eval_cost.py --dataset synthetic
```
