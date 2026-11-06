"""Non-parametric statistics: paired Wilcoxon, Holm correction, Friedman test."""
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np

def paired_wilcoxon_with_holm(method_errors: dict, baseline_errors: dict):
    """Compute paired Wilcoxon signed-rank test and apply Holm step-down correction."""
    pass
