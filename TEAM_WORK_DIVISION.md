# Team Work Division & Interface Protocol

## 1. Overview & Ownership Matrix

| Area | Primary Owner | Module Paths | Key Responsibilities |
|---|---|---|---|
| **Data Ingestion & Harmonization** | **Data Engineer 1 (DE1)** | `src/sohml/data/`<br>`src/sohml/evaluation/transfer.py`<br>`notebooks/01_data_qc.ipynb` | • NASA (raw & cleaned), Oxford (Dataset 1 `C1ch`), Sandia (NCA/NMC 0–100% SOC) loaders<br>• Constant-current segment detection & monotonicity checks<br>• Nominal capacity feature normalization ($Q/Q_{\text{nominal}}$) & label computation ($SOH = 100 \cdot Q / Q_0$)<br>• Resampling to canonical 2 mV uniform voltage grid<br>• Cross-dataset transfer evaluation harness |
| **Feature Engineering & Splits** | **Data Engineer 2 (DE2)** | `src/sohml/features/`<br>`src/sohml/splits/`<br>`src/sohml/baselines/` (two-stage)<br>`src/sohml/evaluation/statistics.py`<br>`notebooks/02_features.ipynb` | • 40+ candidate health indicators ($dQ/dV$, $dT/dV$, Numba-accelerated $SampEn$, kinetics, coupling)<br>• Quantized 10 mV window feature cache (666 valid window pairs)<br>• Outer LOCO and inner GroupKFold splitters with cell sample weights<br>• Two-stage baselines (LASSO, MI, SHAP-RFE feature selection)<br>• Non-parametric statistics (Wilcoxon signed-rank, Holm step-down correction, Friedman + CD diagram) |
| **Model, Optimization & Execution** | **Model Trainer (ML1)** | `src/sohml/models/`<br>`src/sohml/optimization/`<br>`src/sohml/baselines/` (joint)<br>`src/sohml/orchestration/`<br>`src/sohml/evaluation/conformal.py`<br>`notebooks/03_results.ipynb` | • Stacked ensemble (`SVR + RF + XGBoost -> Ridge`) with cell-grouped OOF meta-learner<br>• Model complexity metric ($f_4$)<br>• Mixed decision space encoding/decoding & repair operators<br>• Multi-objective optimizers (NSGA-II, NSGA-III, Hybrid DE-PSO/HHO-GWO) & Hypervolume logging<br>• Utopia knee-point selection with accuracy guard<br>• Joint baselines (Random Search, Optuna TPE) at equal budget $B$<br>• Run orchestrator with checkpoint/resume & Conformal Prediction intervals |

---

## 2. Day-1 Shared Contracts
To ensure zero blocking, all three team members agree on frozen interfaces in `src/sohml/contracts/`:
1. `CanonicalCycleGrid`: Standardized 2 mV resampled CC curves produced by DE1.
2. `FeatureMatrixBundle`: Extracted feature table $(N \times D)$ for any given window $[V_{\text{start}}, V_{\text{end}}]$ produced by DE2.
3. `BaseOptimizer`: Common interface for NSGA-II, Hybrid, Random Search, and Optuna TPE with shared `EvaluationBudgetCounter` maintained by ML1.
4. `synthetic.py`: Deterministic mock battery aging dataset allowing DE2 and ML1 to develop and test their entire pipeline before real data downloads finish.

---

## 3. Milestone Gates
- **Gate 0 (Day 1)**: `schemas.py` frozen; CI pytest passes on synthetic data.
- **Gate 1 (Phase 2)**: Real 2 mV grids generated for NASA, Oxford, and Sandia; window cache precompute completes in <5 min.
- **Gate 2 (Phase 3)**: 1-seed dry run passes the zero-leakage canary test and timing budget check.
- **Gate 3 (Phase 4)**: Full multi-seed runs, cross-dataset transfer, and statistical significance tables generated.
