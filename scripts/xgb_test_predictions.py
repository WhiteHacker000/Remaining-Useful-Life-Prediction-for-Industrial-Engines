"""Phase 1 XGBoost predictions for every test cycle, for the Phase 2 comparison.

Runs in its own process (see src/__init__.py): retrains the tuned XGBoost from
notebook 04 on all training engines and writes outputs/results/xgb_test_predictions.csv.

Usage (from the project root):  python scripts/xgb_test_predictions.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR, RUL_CAP  # noqa: E402
from src.data import add_train_rul, load_cmapss  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.models import make_model  # noqa: E402

if __name__ == "__main__":
    train, test, rul = load_cmapss("FD001")
    train = add_train_rul(train)
    params = json.loads((RESULTS_DIR / "best_params.json").read_text())["xgboost"]
    fb = FeatureBuilder().fit(train)
    X_train, X_test = fb.transform(train), fb.transform(test)
    y = np.minimum(train.loc[X_train.index, "RUL"], RUL_CAP)
    model = make_model("xgboost", **params).fit(X_train[fb.feature_names], y)
    out = pd.DataFrame({"unit": X_test["unit"].to_numpy(), "cycle": X_test["cycle"].to_numpy(),
                        "pred_xgb": np.clip(model.predict(X_test[fb.feature_names]), 0, None)})
    out.to_csv(RESULTS_DIR / "xgb_test_predictions.csv", index=False)
    print(f"saved {len(out):,} predictions")
