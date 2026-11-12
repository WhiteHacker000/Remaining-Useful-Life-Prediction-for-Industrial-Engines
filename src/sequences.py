"""Fixed-length sensor windows for the sequence models (Phase 2).

Each training example is the window of the last `window` cycles of ONE engine,
ending at the cycle we want to predict. Three rules, as in Phase 1:

1. Windows never cross engine boundaries.
2. Windows only look backwards (cycles <= t), so a running engine can be scored.
3. Normalisation (and the engineered features used by the hybrid model) are fitted
   on training engines only.

An engine that has run fewer than `window` cycles is left-padded with its own first
reading, and an extra mask channel marks which time steps are real (1) or padding (0),
so the network can tell a short history from a long, flat one.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from .config import RUL_CAP, SEED, USEFUL_SENSORS
from .features import FeatureBuilder, RegimeScaler


def make_windows(values, units, window, add_mask=True):
    """Build one window per row.

    values : (n_rows, n_channels) array, rows sorted by engine then cycle
    units  : (n_rows,) engine id of each row
    returns: (n_rows, window, n_channels [+1]) float32 array
    """
    values = np.asarray(values, dtype=np.float32)
    units = np.asarray(units)
    n_ch = values.shape[1] + (1 if add_mask else 0)
    out = np.empty((len(values), window, n_ch), dtype=np.float32)

    # row positions where each engine starts and ends (rows are grouped by engine)
    starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
    ends = np.r_[starts[1:], len(units)]
    for s, e in zip(starts, ends):
        seq = values[s:e]
        padded = np.vstack([np.repeat(seq[:1], window - 1, axis=0), seq])
        win = sliding_window_view(padded, window, axis=0)          # (n, ch, window)
        out[s:e, :, : values.shape[1]] = np.moveaxis(win, -1, 1)
        if add_mask:
            mask = np.r_[np.zeros(window - 1, np.float32), np.ones(e - s, np.float32)]
            out[s:e, :, -1] = sliding_window_view(mask, window)
    return out


class SequenceBuilder:
    """Fit on training engines, then turn any set of engines into model inputs.

    transform() returns a dict with
        x     : sensor windows (n, window, 14 sensors + mask)
        f     : standardised Phase 1 engineered features at the window's last cycle (n, 88)
        y     : true RUL (n,)  (NaN-free for train/test engines with labels)
        unit, cycle : identifiers of each row
    """

    def __init__(self, window=30, sensors=USEFUL_SENSORS, n_regimes=1, with_features=True, seed=SEED):
        self.window = window
        self.sensors = list(sensors)
        self.n_regimes = n_regimes
        self.with_features = with_features
        self.seed = seed

    def fit(self, df):
        df = df.sort_values(["unit", "cycle"])
        self.scaler_ = RegimeScaler(self.sensors, self.n_regimes, self.seed).fit(df)
        if self.with_features:
            self.fb_ = FeatureBuilder(self.sensors, self.n_regimes, seed=self.seed).fit(df)
            feats = self.fb_.transform(df)
            self.feature_names = self.fb_.feature_names
            self.f_scaler_ = StandardScaler().fit(feats[self.feature_names])
        return self

    def transform(self, df):
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        z = self.scaler_.transform(df).to_numpy()
        out = {
            "x": make_windows(z, df["unit"].to_numpy(), self.window),
            "unit": df["unit"].to_numpy(),
            "cycle": df["cycle"].to_numpy(),
            "y": df["RUL"].to_numpy(dtype=np.float32) if "RUL" in df else None,
        }
        if self.with_features:
            feats = self.fb_.transform(df)
            out["f"] = self.f_scaler_.transform(feats[self.feature_names]).astype(np.float32)
        return out


def capped(y, cap=RUL_CAP):
    return np.minimum(y, cap).astype(np.float32)


def subset(data, mask):
    """Rows of a transform() dict selected by a boolean mask."""
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
