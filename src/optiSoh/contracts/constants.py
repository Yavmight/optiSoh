"""Single Source of Truth for voltage limits, grid resolutions, and window parameters."""

# Canonical 2 mV voltage grid parameters
DEFAULT_V_GRID_MIN: float = 3.500
DEFAULT_V_GRID_MAX: float = 4.180
DEFAULT_V_GRID_STEP: float = 0.002  # 2 mV step

# Partial-charging window search boundaries
WINDOW_V_MIN: float = 3.60
WINDOW_V_MAX: float = 4.10
WINDOW_MIN_WIDTH: float = 0.15  # 150 mV minimum charging window width
WINDOW_QUANTIZATION_STEP_MV: int = 10  # 10 mV cache quantization step

# Decision vector continuous parameter ranges
WINDOW_START_MIN: float = 3.60
WINDOW_START_MAX: float = 3.95
WINDOW_END_MIN: float = 3.75
WINDOW_END_MAX: float = 4.10
