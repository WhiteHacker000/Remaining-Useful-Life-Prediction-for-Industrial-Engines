"""Model definitions, engine-level cross-validation and hyperparameter search."""

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import N_FOLDS, RUL_CAP, SEED
from .evaluation import rmse
from .features import FeatureBuilder

MODEL_NAMES = {
    "mean": "Mean baseline",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

DEFAULT_PARAMS = {
    "mean": {},
    "elastic_net": {"alpha": 0.1, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.33},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4,
                "subsample": 0.8, "colsample_bytree": 0.8},
}

SEARCH_SPACES = {
    "elastic_net": {"alpha": loguniform(1e-2, 10), "l1_ratio": uniform(0.05, 0.95)},
    "random_forest": {"n_estimators": [150, 250], "max_depth": [None, 8, 12, 16],
                      "min_samples_leaf": randint(1, 30), "max_features": uniform(0.1, 0.6)},
    "xgboost": {"n_estimators": randint(150, 800), "learning_rate": loguniform(0.01, 0.2),
                "max_depth": randint(2, 7), "min_child_weight": randint(1, 20),
                "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.6),
                "reg_lambda": loguniform(0.1, 20)},
}


def make_model(name, **params):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "elastic_net":
        # Linear models need standardised inputs; the scaler is fitted inside each fold.
        return make_pipeline(StandardScaler(), ElasticNet(max_iter=5000, random_state=SEED, **params))
    if name == "random_forest":
        return RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
    if name == "xgboost":
        return XGBRegressor(tree_method="hist", n_jobs=-1, random_state=SEED, **params)
    raise ValueError(f"unknown model {name}")


def engine_folds(units, n_folds=N_FOLDS, seed=SEED):
    """Split engine ids (not rows) into folds."""
    units = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    return [np.sort(f) for f in np.array_split(units, n_folds)]


def build_cv_folds(train, n_folds=N_FOLDS, seed=SEED, **builder_kwargs):
    """Engineer features separately for every fold.

    The FeatureBuilder is fitted on the training engines of the fold only, so the
    validation engines never influence normalisation or the health index.
    """
    folds = []
    for val_units in engine_folds(train["unit"], n_folds, seed):
        is_val = train["unit"].isin(val_units)
        tr, va = train[~is_val], train[is_val]
        fb = FeatureBuilder(**builder_kwargs).fit(tr)
        X_tr, X_va = fb.transform(tr), fb.transform(va)
        folds.append({
            "X_tr": X_tr, "y_tr": tr.loc[X_tr.index, "RUL"].to_numpy(),
            "X_va": X_va, "y_va": va.loc[X_va.index, "RUL"].to_numpy(),
            "units_va": X_va["unit"].to_numpy(),
        })
    return folds, fb.groups


def cross_validate(name, folds, features, params=None, cap=RUL_CAP, return_oof=False):
    """Train on each fold, return per-fold RMSE against the capped target.

    cap=None trains on the raw (linear) RUL instead.
    """
    params = DEFAULT_PARAMS[name] if params is None else params
    scores, oof = [], []
    for f in folds:
        y_tr = f["y_tr"] if cap is None else np.minimum(f["y_tr"], cap)
        model = make_model(name, **params).fit(f["X_tr"][features], y_tr)
        pred = np.clip(model.predict(f["X_va"][features]), 0, None)
        y_eval = f["y_va"] if cap is None else np.minimum(f["y_va"], cap)
        scores.append(rmse(y_eval, pred))
        if return_oof:
            oof.append(pd.DataFrame({"unit": f["units_va"], "cycle": f["X_va"]["cycle"].to_numpy(),
                                     "true_rul": f["y_va"], "pred": pred}))
    scores = np.array(scores)
    if return_oof:
        return scores, pd.concat(oof, ignore_index=True)
    return scores


def random_search(name, folds, features, n_iter=20, seed=SEED):
    """Random search over SEARCH_SPACES[name], scored by mean CV RMSE."""
    rows = []
    for params in ParameterSampler(SEARCH_SPACES[name], n_iter=n_iter, random_state=seed):
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        scores = cross_validate(name, folds, features, params)
        rows.append({**params, "cv_rmse": scores.mean(), "cv_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
