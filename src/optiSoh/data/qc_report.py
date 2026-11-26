"""Generates data quality control reports and window coverage summaries."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ..contracts.schemas import DatasetBundle

DEFAULT_AUDIT_WINDOWS: List[Tuple[float, float]] = [
    (3.60, 4.10),
    (3.75, 4.10),
    (3.85, 4.10),
    (3.90, 4.10),
]


def _window_coverage(cycles, v_start: float, v_end: float) -> Tuple[int, float]:
    if not cycles:
        return 0, 0.0
    covered = 0
    for c in cycles:
        mask = (c.v_grid >= v_start - 1e-6) & (c.v_grid <= v_end + 1e-6)
        if np.any(mask) and np.all(np.isfinite(c.q_norm[mask])) and np.all(np.isfinite(c.dq_dv[mask])):
            covered += 1
    return covered, 100.0 * covered / len(cycles)


def generate_qc_markdown_report(
    bundle: DatasetBundle,
    output_path: Union[str, Path],
    windows: List[Tuple[float, float]] = DEFAULT_AUDIT_WINDOWS,
) -> str:
    """Write a Markdown summary table of per-cell SOH trajectories, SOH bands, and [V_start, V_end] coverage."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    eol_thresh = float(bundle.metadata.get("eol_soh_threshold", 80.0))
    eol_int = int(round(eol_thresh))
    cycle_unit = str(bundle.metadata.get("cycle_index_unit", "cycles"))
    total_below_eol = sum(1 for c in bundle.cycles if c.soh_true < eol_thresh)
    pct_below_eol = (100.0 * total_below_eol / len(bundle.cycles)) if bundle.cycles else 0.0

    unique_caps = sorted({round(float(bundle.get_nominal_capacity(cid)), 4) for cid in bundle.cell_ids})
    unique_temps = sorted({round(float(bundle.get_ambient_temp(cid)), 1) for cid in bundle.cell_ids})
    if len(unique_caps) > 1:
        cap_summary = ", ".join(f"{c:g} Ah" for c in unique_caps) + " (heterogeneous per-cell)"
    elif len(unique_caps) == 1:
        cap_summary = f"{unique_caps[0]:g} Ah"
    else:
        cap_summary = f"{bundle.nominal_capacity} Ah"

    if len(unique_temps) > 1:
        temp_summary = ", ".join(f"{t:g} °C" for t in unique_temps) + " (heterogeneous per-cell)"
    elif len(unique_temps) == 1:
        temp_summary = f"{unique_temps[0]:g} °C"
    else:
        temp_summary = f"{bundle.ambient_temp_c} °C"

    lines = [
        f"# Data Quality Control Report: `{bundle.name}`",
        "",
        f"- **Cells included**: {len(bundle.cell_ids)} ({', '.join(bundle.cell_ids)})",
        f"- **Total harmonized cycles**: {len(bundle.cycles)} (`cycle_index` unit: `{cycle_unit}`)",
        f"- **Nominal capacity**: {cap_summary}",
        f"- **Ambient temperature**: {temp_summary}",
        f"- **EOL SOH threshold**: {eol_thresh:.1f}% ({total_below_eol}/{len(bundle.cycles)} cycles, {pct_below_eol:.1f}% below EOL — kept for per-band reporting)",
        "",
        "## Per-Cell Summary & Partial-Window Coverage",
        "",
    ]

    win_headers = [f"[{s:.2f}, {e:.2f}] V" for s, e in windows]
    header = (
        "| Cell | Cycles | Regen | Outliers | SOH Start (%) | SOH Min (%) | SOH End (%) | V_start (p50 / p95) | V_end (min) | "
        + " | ".join(win_headers)
        + " |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---|---:|" + "|".join(["---:"] * len(windows)) + "|"
    lines.extend([header, sep])

    band_rows = []
    for cid in bundle.cell_ids:
        cell_cycles = [c for c in bundle.cycles if c.cell_id == cid]
        if not cell_cycles:
            continue
        sohs = [c.soh_true for c in cell_cycles]
        n_regen = sum(1 for c in cell_cycles if getattr(c, "is_regeneration", False))
        n_outliers = sum(1 for c in cell_cycles if getattr(c, "is_label_outlier", False))
        v_starts = [float(c.v_grid[np.isfinite(c.q_norm)][0]) for c in cell_cycles if np.any(np.isfinite(c.q_norm))]
        v_ends = [float(c.v_grid[np.isfinite(c.q_norm)][-1]) for c in cell_cycles if np.any(np.isfinite(c.q_norm))]

        v_s_p50 = float(np.median(v_starts)) if v_starts else float("nan")
        v_s_p95 = float(np.quantile(v_starts, 0.95)) if v_starts else float("nan")
        v_e_min = float(np.min(v_ends)) if v_ends else float("nan")

        cov_cells = []
        for s, e in windows:
            cnt, pct = _window_coverage(cell_cycles, s, e)
            cov_cells.append(f"{cnt}/{len(cell_cycles)} ({pct:.1f}%)")

        row = (
            f"| {cid} | {len(cell_cycles)} | {n_regen} | {n_outliers} | {sohs[0]:.2f} | {min(sohs):.2f} | {sohs[-1]:.2f} | "
            f"{v_s_p50:.3f} / {v_s_p95:.3f} V | {v_e_min:.3f} V | " + " | ".join(cov_cells) + " |"
        )
        lines.append(row)

        n_gt90 = sum(1 for s in sohs if s > 90.0)
        n_mid = sum(1 for s in sohs if eol_thresh <= s <= 90.0)
        n_lt_eol = sum(1 for s in sohs if s < eol_thresh)
        n_tot = len(sohs)
        band_rows.append(
            f"| {cid} | {n_gt90} ({100.0 * n_gt90 / n_tot:.1f}%) | "
            f"{n_mid} ({100.0 * n_mid / n_tot:.1f}%) | "
            f"{n_lt_eol} ({100.0 * n_lt_eol / n_tot:.1f}%) |"
        )

    # Build data-driven coverage note from actual observed coverage on [3.60, 4.10] V
    full_win_pcts = []
    v_start_p95s = []
    for cid in bundle.cell_ids:
        cell_cycles = [c for c in bundle.cycles if c.cell_id == cid]
        if not cell_cycles:
            continue
        _, pct = _window_coverage(cell_cycles, 3.60, 4.10)
        full_win_pcts.append(pct)
        v_starts = [float(c.v_grid[np.isfinite(c.q_norm)][0]) for c in cell_cycles if np.any(np.isfinite(c.q_norm))]
        if v_starts:
            v_start_p95s.append(float(np.quantile(v_starts, 0.95)))

    if full_win_pcts and min(full_win_pcts) >= 95.0:
        coverage_note = (
            f"Points outside a cycle's measured CC voltage span are stored as `NaN` (zero extrapolation). "
            f"All cells achieve >=95% coverage on `[3.60, 4.10] V` (min cell coverage: {min(full_win_pcts):.1f}%)."
        )
    elif full_win_pcts:
        v95_lo, v95_hi = min(v_start_p95s), max(v_start_p95s)
        coverage_note = (
            f"Points outside a cycle's measured CC voltage span are stored as `NaN` (zero extrapolation). "
            f"Coverage on `[3.60, 4.10] V` ranges from {min(full_win_pcts):.1f}% to {max(full_win_pcts):.1f}% across cells "
            f"(95th-percentile initial CC voltage: {v95_lo:.3f}–{v95_hi:.3f} V); downstream optimization must enforce per-cell coverage guards."
        )
    else:
        coverage_note = "Points outside a cycle's measured CC voltage span are stored as `NaN` (zero extrapolation)."

    lines.extend(
        [
            "",
            f"## Per-Cell SOH Band Distribution (EOL = {eol_int}%)",
            "",
            f"| Cell | > 90% SOH | {eol_int}–90% SOH | < {eol_int}% SOH (Below EOL) |",
            "|---|---:|---:|---:|",
            *band_rows,
            "",
            f"> **Note**: {coverage_note}",
        ]
    )
    report = "\n".join(lines) + "\n"
    out.write_text(report, encoding="utf-8")
    return report
