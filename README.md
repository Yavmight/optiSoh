# optiSoh: Window-Adaptive Co-Optimization for Battery SOH Regression

A leakage-safe machine learning and optimization framework for **Window-Adaptive, Multi-Objective Metaheuristic Co-Optimization of Features and Hyperparameters for Partial-Charging Battery State-of-Health (SOH) Regression**.

---

## 1. Project Aim & Motivation
Most laboratory battery studies rely on full 0–100% charge cycles to estimate State of Health (SOH). In real-world electric vehicles (EVs), however, batteries almost always operate and charge within **partial State-of-Charge (SOC) windows**.

This project investigates a central hypothesis: **does jointly learning the partial-charging voltage window $[V_{\text{start}}, V_{\text{end}}]$, an active feature mask, and regressor hyperparameters via Pareto multi-objective metaheuristics outperform equally tuned step-by-step baselines under strict Leave-One-Cell-Out (LOCO) evaluation and cross-dataset transfer?**

---

## 2. Methodology & Key Novelties
1. **Searchable Partial-Charging Window**: Continuous window bounds $[V_{\text{start}}, V_{\text{end}}]$ (within 3.60 V–4.10 V, minimum width $\ge 0.15\text{ V}$) are optimized directly inside the decision vector alongside feature selection and regressor hyperparameters.
2. **4-Objective Pareto Optimization**: Replaces arbitrary scalarized weighting with non-dominated Pareto search over:
   - $f_1$: Mean cross-cell RMSE
   - $f_2$: Worst-cell RMSE (cross-cell robustness)
   - $f_3$: Feature fraction ($N_{\text{selected}} / N_{\text{total}}$)
   - $f_4$: Model complexity (support vector fraction and normalized tree depths)
3. **Leakage-Safe Nested Evaluation**: The optimizer operates strictly on training cells with inner cell-disjoint validation folds. Outer test cells are evaluated only once on the selected configuration to prevent validation-set overfitting.
4. **Guarded Knee-Point Selection**: Automatically selects the best-compromise solution on the Pareto front, filtering out degenerate underfitted corners.

---

## 3. End-to-End Pipeline
```text
[Raw Battery Cycling Data (NASA / Oxford / Sandia)]
                         │
                         ▼
  1. Ingestion & QC (Constant-Current Plateau Detection, Monotonicity Check)
                         │
                         ▼
  2. Physics Normalization (Q/Q_nominal) & 2 mV Uniform Voltage-Grid Resampling
                         │
                         ▼
  3. Window-Adaptive Feature Extraction (40+ IC, DT, Entropy, Kinetics features)
     & 10 mV Fast Quantized Cache (<1 ms lookups)
                         │
                         ▼
  4. Leakage-Safe Outer LOCO & Inner Cell-Disjoint Splitters
                         │
                         ▼
  5. Pareto Multi-Objective Search (NSGA-II / NSGA-III / Hybrid ADE-PSO)
                         │
                         ▼
  6. Guarded Knee-Point Selection ➔ Final Retraining on Train Cells
                         │
                         ▼
  7. Held-Out Cell Evaluation, Conformal Prediction Intervals & Wilcoxon-Holm Tests
```

---

## 4. Models, Baselines & Datasets

### Models
- **Primary Stacked Ensemble**: Fixed-structure stack combining **Support Vector Regression (RBF SVR)**, **Random Forest**, and **XGBoost**, blended via a **Ridge** meta-learner trained strictly on cell-grouped out-of-fold (OOF) predictions.
- **Optimizers**: NSGA-II, NSGA-III, and Hybrid Multi-Objective Adaptive DE-PSO (`MO-ADEPSO`).

### Equal-Budget Baselines
- **Two-Stage Baselines**: LASSO ($L_1$), Mutual Information (MI), and SHAP-based Recursive Feature Elimination (SHAP-RFE), each paired with an Optuna-tuned stacked ensemble.
- **Joint Baselines**: Random Search and Optuna TPE / MOTPE matched to the exact same fitness evaluation budget $B$.
- **Reference Models**: Fixed Default Ensemble (full window) and Gaussian Process Regression (GPR).

### Datasets
- **NASA PCoE Dataset**: Cylindrical 18650 cells cycled at 24 °C (1.5 A CC charging).
- **Oxford Battery Degradation Dataset 1**: Kokam pouch cells cycled at 40 °C (1C CC characterization curves in `C1ch`).
- **Sandia National Laboratories (SNL Preger 2020)**: Commercial NCA and NMC 18650 cells cycled over 0–100% SOC with thermocouple temperature logging.

---

## 5. Team Work Division & Codebase Map
The codebase is structured around a 3-person non-blocking ownership protocol:

| Role | Core Ownership | Key Directories |
|---|---|---|
| **Data Engineer 1 (DE1)** | Ingestion, CC quality control, nominal normalization, 2 mV grid harmonization, cross-dataset transfer harness | `src/optiSoh/data/`, `evaluation/transfer.py`, `notebooks/01_data_qc.ipynb` |
| **Data Engineer 2 (DE2)** | 40+ candidate features (`dQ/dV`, `dT/dV`, Numba $SampEn$), 10 mV cache, LOCO splitters, two-stage baselines, Wilcoxon/Holm stats | `src/optiSoh/features/`, `splits/`, `baselines/`, `evaluation/statistics.py` |
| **Model Trainer (ML1)** | Stacked ensemble, multi-objective search (NSGA-II/III, Hybrid), knee selection, joint baselines, run orchestration, conformal UQ | `src/optiSoh/models/`, `optimization/`, `orchestration/`, `evaluation/conformal.py` |

*See `docs/data_contracts.md` and `docs/experiment_protocol.md` for interface contracts and protocol specifications.*

---

## 6. Quickstart & Usage

### Setup Virtual Environment
```powershell
# Navigate to project root
cd "optiSoh"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package with development dependencies
pip install -e ".[dev]"
pre-commit install
```

### Run Automated Tests & Benchmarks
```powershell
# Run unit and zero-leakage canary tests
pytest tests -v

# Benchmark fitness evaluation cost on local hardware
python scripts/benchmark_eval_cost.py --dataset synthetic

# Run smoke test on synthetic battery cycles
python scripts/run_experiment.py --mode smoke --budget 24 --seeds 2
```
