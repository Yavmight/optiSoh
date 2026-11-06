# Data Contracts & Schemas

To ensure zero leakage and enable seamless parallel work across DE1, DE2, and ML1:

1. **Nominal Capacity Normalization (Label Leakage Guard)**:
   - Features MUST be normalized by the cell's **nominal rated capacity** ($Q_{\text{nominal}}$, e.g. 2.0 Ah for NASA, 0.74 Ah for Oxford).
   - The SOH target label is computed from the cell's **measured initial capacity** $Q_0$: $\text{SOH} = 100 \times (Q_k / Q_0)$.
   - Normalizing features with measured $Q_0$ leaks target information to the test set and is strictly prohibited.

2. **Canonical 2 mV Voltage Grid**:
   - Monotonic charging voltage: $V \in [3.500, 4.180]$ with step $\Delta V = 0.002$ V.
   - Channels: $Q_{\text{norm}}(V)$, $dQ_{\text{norm}}/dV$, $\Delta T(V)$, $dT/dV$, $\tau(V)$ (C-rate normalized time).

3. **Forbidden Features**:
   - Explicitly banned: `cycle_index`, `time_elapsed`, `cumulative_ah`, `cell_id`, `dataset_name`.
