"""Reproduce the Phase 1 tuned XGBoost result on the official FD001 test set.

Run from the repository root with:
    python -m scripts.reproduce_phase1_baseline
"""

import json
import sys
from pathlib import Path
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas
import sklearn

from src.config import RUL_CAP, SEED  # noqa: E402
from src.data import add_test_rul, add_train_rul, load_cmapss  # noqa: E402
from src.evaluation import mae, nasa_score, rmse  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.models import make_model  # noqa: E402


def main():
    results_dir = ROOT / "outputs" / "results"
    best_params = json.loads((results_dir / "best_params.json").read_text())
    train_raw, test_raw, test_rul = load_cmapss("FD001")
    train = add_train_rul(train_raw)
    test = add_test_rul(test_raw, test_rul)

    builder = FeatureBuilder().fit(train)
    train_features = builder.transform(train)
    test_features = builder.transform(test)
    train_target = np.minimum(
        train.loc[train_features.index, "RUL"].to_numpy(), RUL_CAP
    )

    model = make_model("xgboost", **best_params["xgboost"])
    model.fit(train_features[builder.feature_names], train_target)

    last_indices = test.groupby("unit")["cycle"].idxmax().to_numpy()
    final_features = test_features.loc[last_indices]
    y_true = test.loc[last_indices, "RUL"].to_numpy()
    y_pred = np.clip(model.predict(final_features[builder.feature_names]), 0, None)

    result = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation": "Official test engines; one prediction at each engine's last observed cycle.",
        "selection": "Saved best parameters selected by five-fold engine-level CV; official test is not used for selection.",
        "target_cap": RUL_CAP,
        "seed": SEED,
        "train_engines": int(train["unit"].nunique()),
        "official_test_engines": int(test["unit"].nunique()),
        "test_predictions": int(len(y_true)),
        "feature_count": int(len(builder.feature_names)),
        "model": "XGBoost",
        "model_parameters": best_params["xgboost"],
        "metrics": {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "NASA_score": nasa_score(y_true, y_pred),
        },
        "environment": {
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": version("xgboost"),
        },
    }

    output = results_dir / "phase1_baseline_reproduced.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
