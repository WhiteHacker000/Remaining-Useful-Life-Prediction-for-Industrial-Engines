"""Degradation feature engineering.

Two rules are followed everywhere in this module:

1. Causal features. A feature at cycle t only uses cycles <= t of the same engine,
   so the same code can be applied to a test engine that is still running.
2. Fit on training engines only. Anything that learns from data (regime clusters,
   sensor means/stds, the PCA health index) is fitted inside `FeatureBuilder.fit`
   and then reused unchanged for validation or test engines.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import SEED, SETTING_COLS, USEFUL_SENSORS


# ----------------------------------------------------------------------------
# Rolling helpers (all grouped by engine)
# ----------------------------------------------------------------------------

def rolling_mean(df, col, window):
    return (
        df.groupby("unit")[col]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_std(df, col, window):
    # Wait for half a window of history; a std from 2-3 noisy points is mostly noise.
    # Before that the feature is 0 ("no variation measured yet").
    std = (
        df.groupby("unit")[col]
        .rolling(window, min_periods=max(2, window // 2))
        .std()
        .reset_index(level=0, drop=True)
    )
    return std.fillna(0.0)


def rolling_slope(df, col, window):
    """Least-squares slope of `col` against cycle over the last `window` cycles.

    slope = cov(t, x) / var(t), computed from rolling means of t, x, t*x and t^2.
    Needs half a window of history (at least 3 points); before that the slope is 0.
    """
    t = df["cycle"].astype(float)
    tmp = pd.DataFrame({"unit": df["unit"], "t": t, "x": df[col], "tx": t * df[col], "tt": t * t})
    m = (
        tmp.groupby("unit")[["t", "x", "tx", "tt"]]
        .rolling(window, min_periods=max(3, window // 2))
        .mean()
        .reset_index(level=0, drop=True)
    )
    slope = (m["tx"] - m["t"] * m["x"]) / (m["tt"] - m["t"] ** 2)
    return slope.reindex(df.index).fillna(0.0)


def baseline_mean(df, col, n_cycles=10):
    """Mean of the engine's first `n_cycles` cycles (expanding mean before that)."""
    early = df[col].where(df["cycle"] <= n_cycles)
    return (
        early.groupby(df["unit"])
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )


# ----------------------------------------------------------------------------
# Operating-regime normalisation
# ----------------------------------------------------------------------------

class RegimeScaler:
    """Cluster the operating settings into regimes and z-score each sensor per regime.

    FD001 has a single operating condition, so n_regimes=1 reduces to an ordinary
    z-score. FD002/FD004 have six conditions and need n_regimes=6.
    """

    def __init__(self, sensors=USEFUL_SENSORS, n_regimes=1, seed=SEED):
        self.sensors = list(sensors)
        self.n_regimes = n_regimes
        self.seed = seed

    def fit(self, df):
        if self.n_regimes > 1:
            self.setting_scaler_ = StandardScaler().fit(df[SETTING_COLS])
            self.kmeans_ = KMeans(self.n_regimes, n_init=10, random_state=self.seed)
            self.kmeans_.fit(self.setting_scaler_.transform(df[SETTING_COLS]))
        regimes = self.regimes(df)
        grouped = df[self.sensors].groupby(regimes)
        self.mean_ = grouped.mean()
        self.std_ = grouped.std().replace(0, 1.0)
        return self

    def regimes(self, df):
        if self.n_regimes == 1:
            return pd.Series(0, index=df.index, name="regime")
        labels = self.kmeans_.predict(self.setting_scaler_.transform(df[SETTING_COLS]))
        return pd.Series(labels, index=df.index, name="regime")

    def transform(self, df):
        r = self.regimes(df)
        mean = self.mean_.loc[r].to_numpy()
        std = self.std_.loc[r].to_numpy()
        return pd.DataFrame((df[self.sensors].to_numpy() - mean) / std,
                            index=df.index, columns=self.sensors)


# ----------------------------------------------------------------------------
# Full feature pipeline
# ----------------------------------------------------------------------------

class FeatureBuilder:
    """Turns raw C-MAPSS rows into the engineered feature table.

    Feature groups (names are stored in `self.groups` after transform):
        sensors       regime-normalised sensor values
        cycle         current operating cycle
        rolling       rolling mean and rolling std over a short window
        slope         rolling least-squares slope over a short and a long window
        baseline_dev  smoothed value minus the engine's own early-life baseline
        health        1-D PCA health index, its smoothed value and long-window slope
        regime        regime id (only when n_regimes > 1)
    """

    def __init__(self, sensors=USEFUL_SENSORS, n_regimes=1, short_window=10,
                 long_window=30, baseline_cycles=10, seed=SEED):
        self.sensors = list(sensors)
        self.n_regimes = n_regimes
        self.short_window = short_window
        self.long_window = long_window
        self.baseline_cycles = baseline_cycles
        self.seed = seed

    def fit(self, df):
        self.scaler_ = RegimeScaler(self.sensors, self.n_regimes, self.seed).fit(df)
        z = self.scaler_.transform(df)
        self.pca_ = PCA(n_components=1, random_state=self.seed).fit(z)
        # PCA's sign is arbitrary. Orient it so the index falls as the engine ages.
        hi = self.pca_.transform(z)[:, 0]
        self.hi_sign_ = -1.0 if np.corrcoef(hi, df["cycle"])[0, 1] > 0 else 1.0
        return self

    def transform(self, df):
        df = df.sort_values(["unit", "cycle"])
        w, W = self.short_window, self.long_window
        z = self.scaler_.transform(df)
        work = pd.concat([df[["unit", "cycle"]], z], axis=1)

        feats = {"cycle": df["cycle"].astype(float)}
        groups = {k: [] for k in ["sensors", "cycle", "rolling", "slope", "baseline_dev", "health"]}
        groups["cycle"] = ["cycle"]

        for s in self.sensors:
            feats[s] = z[s]
            groups["sensors"].append(s)

            mean_w = rolling_mean(work, s, w)
            feats[f"{s}_mean{w}"] = mean_w
            feats[f"{s}_std{w}"] = rolling_std(work, s, w)
            groups["rolling"] += [f"{s}_mean{w}", f"{s}_std{w}"]

            feats[f"{s}_slope{w}"] = rolling_slope(work, s, w)
            feats[f"{s}_slope{W}"] = rolling_slope(work, s, W)
            groups["slope"] += [f"{s}_slope{w}", f"{s}_slope{W}"]

            feats[f"{s}_dev"] = mean_w - baseline_mean(work, s, self.baseline_cycles)
            groups["baseline_dev"].append(f"{s}_dev")

        work["health"] = self.hi_sign_ * self.pca_.transform(z)[:, 0]
        feats["health"] = work["health"]
        feats[f"health_mean{w}"] = rolling_mean(work, "health", w)
        feats[f"health_slope{W}"] = rolling_slope(work, "health", W)
        groups["health"] = ["health", f"health_mean{w}", f"health_slope{W}"]

        if self.n_regimes > 1:
            feats["regime"] = self.scaler_.regimes(df)
            groups["regime"] = ["regime"]

        self.groups = groups
        self.feature_names = [c for g in groups.values() for c in g]
        # "unit" is kept for grouping; it is never used as a model input.
        return pd.concat([df[["unit"]], pd.DataFrame(feats, index=df.index)], axis=1)

    def fit_transform(self, df):
        return self.fit(df).transform(df)
