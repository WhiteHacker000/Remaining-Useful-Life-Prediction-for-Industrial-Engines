"""Run engine-holdout LSTM/window/regularization experiments.

Run from the repository root with:
    python -m scripts.run_phase2_validation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.data import add_train_rul, load_train_cmapss  # noqa: E402
from src.experiments import run_phase2_validation  # noqa: E402


if __name__ == "__main__":
    train = add_train_rul(load_train_cmapss("FD001"))
    result = run_phase2_validation(
        train,
        output_dir=RESULTS_DIR / "phase2",
        best_params_path=RESULTS_DIR / "best_params.json",
    )
    print(f"\nSelected: {result['selected_experiment']['name']}")
