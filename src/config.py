"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

SEED = 42
N_FOLDS = 5

# Piecewise-linear target: RUL is clipped at this value during training.
# The choice is justified in notebooks 02 and 04.
RUL_CAP = 125

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensor descriptions from Saxena et al. (2008), Table 2.
SENSOR_INFO = {
    "s_1": ("T2", "Total temperature at fan inlet", "°R"),
    "s_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s_5": ("P2", "Pressure at fan inlet", "psia"),
    "s_6": ("P15", "Total pressure in bypass duct", "psia"),
    "s_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s_8": ("Nf", "Physical fan speed", "rpm"),
    "s_9": ("Nc", "Physical core speed", "rpm"),
    "s_10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s_13": ("NRf", "Corrected fan speed", "rpm"),
    "s_14": ("NRc", "Corrected core speed", "rpm"),
    "s_15": ("BPR", "Bypass ratio", "-"),
    "s_16": ("farB", "Burner fuel-air ratio", "-"),
    "s_17": ("htBleed", "Bleed enthalpy", "-"),
    "s_18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s_19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s_21": ("W32", "LPT coolant bleed", "lbm/s"),
}

# The 14 sensors kept after the inspection in notebook 01/02.
# Dropped: s_1, s_5, s_10, s_16, s_18, s_19 (constant) and s_6 (two values only).
USEFUL_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11",
    "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21",
]
