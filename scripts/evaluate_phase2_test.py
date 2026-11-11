"""Fit the validation-selected LSTM on all FD001 training engines and evaluate once.

Run from the repository root with:
    python -m scripts.evaluate_phase2_test
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import RUL_CAP, RESULTS_DIR, SEED  # noqa: E402
from src.data import add_test_rul, add_train_rul, load_cmapss  # noqa: E402
from src.evaluation import (  # noqa: E402
    bootstrap_rmse,
    errors_by_stage,
    mae,
    nasa_score,
    rmse,
    stage_of,
)
from src.sequence_data import EngineStandardizer, make_sequence_windows  # noqa: E402
from src.training import TrainingConfig, predict_lstm, train_lstm  # noqa: E402


def main():
    phase2_dir = RESULTS_DIR / "phase2"
    selection = json.loads((phase2_dir / "selection.json").read_text())
    selected = selection["selected_experiment"]
    train_raw, test_raw, test_rul = load_cmapss("FD001")
    train = add_train_rul(train_raw)
    test = add_test_rul(test_raw, test_rul)

    standardizer = EngineStandardizer().fit(train, train["unit"].unique())
    normalized_train = standardizer.transform(train)
    normalized_test = standardizer.transform(test)
    train_windows = make_sequence_windows(
        normalized_train,
        window_size=selected["window_size"],
        target_cap=RUL_CAP,
    )
    test_windows = make_sequence_windows(
        normalized_test,
        window_size=selected["window_size"],
        target_cap=None,
        last_only=True,
    )

    training_config = TrainingConfig(**selected["training_config"])
    training_config = replace(training_config, epochs=selected["best_epoch"])
    model_dir = ROOT / "models" / "phase2"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "lstm_fd001.pt"
    result = train_lstm(
        train_windows,
        None,
        checkpoint_path,
        model_config=selected["model_config"],
        config=training_config,
    )
    prediction = np.clip(
        predict_lstm(
            result.model,
            test_windows,
            batch_size=training_config.batch_size,
            device=result.device,
        ),
        0,
        None,
    )
    y_true = test_windows.y.astype(np.float64)
    y_capped = np.minimum(y_true, RUL_CAP)

    test_predictions = pd.DataFrame(
        {
            "unit": test_windows.unit,
            "last_observed_cycle": test_windows.cycle,
            "true_rul": y_true,
            "capped_rul": y_capped,
            "predicted_rul": prediction,
            "error_pred_minus_true": prediction - y_true,
            "absolute_error": np.abs(prediction - y_true),
        }
    )
    test_predictions["stage"] = stage_of(y_true)

    baseline_predictions = pd.read_csv(
        RESULTS_DIR / "test_predictions_last_cycle.csv"
    ).sort_values("unit")
    if not np.array_equal(baseline_predictions["unit"].to_numpy(), test_windows.unit):
        raise ValueError("Phase 1 and Phase 2 test engines are not aligned")
    if not np.array_equal(baseline_predictions["true_rul"].to_numpy(), y_true):
        raise ValueError("Phase 1 and Phase 2 official test labels do not match")
    xgb_prediction = baseline_predictions["pred_xgboost"].to_numpy()

    ci_low, ci_high = bootstrap_rmse(y_true, prediction, seed=SEED)
    metrics = {
        "dataset": "NASA C-MAPSS FD001",
        "test_engines": int(len(y_true)),
        "evaluation": "One prediction at each official test engine's last observed cycle.",
        "selection_protocol": "Configuration selected only on a seeded 20-engine validation holdout; no test metric influenced selection.",
        "selected_experiment": selected,
        "final_fit_epochs": result.best_epoch,
        "seed": SEED,
        "window_size": selected["window_size"],
        "input_sensors": standardizer.sensor_columns,
        "target_cap_for_training": RUL_CAP,
        "metrics_true_RUL": {
            "RMSE": rmse(y_true, prediction),
            "MAE": mae(y_true, prediction),
            "NASA_score": nasa_score(y_true, prediction),
            "RMSE_95_CI": [ci_low, ci_high],
        },
        "metrics_capped_RUL": {
            "RMSE": rmse(y_capped, prediction),
            "MAE": mae(y_capped, prediction),
            "NASA_score": nasa_score(y_capped, prediction),
        },
        "phase1_xgboost_same_test_population": {
            "RMSE": rmse(y_true, xgb_prediction),
            "MAE": mae(y_true, xgb_prediction),
            "NASA_score": nasa_score(y_true, xgb_prediction),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "history": str(result.history_path.relative_to(ROOT)),
            "preprocessing": "models/phase2/lstm_fd001_preprocessing.json",
        },
    }

    preprocessor = {
        "dataset": "NASA C-MAPSS FD001",
        "sensor_columns": standardizer.sensor_columns,
        "mean": standardizer.scaler_.mean_.tolist(),
        "scale": standardizer.scaler_.scale_.tolist(),
        "window_size": selected["window_size"],
        "padding": "Repeat the first observed normalized sensor vector on the left.",
        "target_cap": RUL_CAP,
        "seed": SEED,
    }
    (model_dir / "lstm_fd001_preprocessing.json").write_text(
        json.dumps(preprocessor, indent=2) + "\n"
    )
    (phase2_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    test_predictions.to_csv(phase2_dir / "test_predictions.csv", index=False)

    stage_tables = []
    for model_name, model_prediction in [
        ("XGBoost", xgb_prediction),
        ("LSTM", prediction),
    ]:
        stage_table = errors_by_stage(y_true, model_prediction, stage_rul=y_true)
        stage_tables.append(
            stage_table.reset_index().assign(model=model_name)
        )
    stage_results = pd.concat(stage_tables, ignore_index=True)
    stage_results.to_csv(phase2_dir / "test_error_by_stage.csv", index=False)

    phase1_comparison = pd.read_csv(
        RESULTS_DIR / "test_final_models.csv"
    ).rename(columns={"Score": "NASA_score"})
    lstm_comparison = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "variant": selected["name"],
                "RMSE": metrics["metrics_true_RUL"]["RMSE"],
                "MAE": metrics["metrics_true_RUL"]["MAE"],
                "NASA_score": metrics["metrics_true_RUL"]["NASA_score"],
                "RMSE 95% CI": f"{ci_low:.1f} - {ci_high:.1f}",
                "RMSE vs capped truth": metrics["metrics_capped_RUL"]["RMSE"],
            }
        ]
    )
    comparison = pd.concat(
        [phase1_comparison, lstm_comparison], ignore_index=True
    )
    comparison.to_csv(phase2_dir / "model_comparison.csv", index=False)

    figure_dir = ROOT / "outputs" / "figures" / "phase2"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, xgb_prediction, alpha=0.7, label="Phase 1 XGBoost")
    ax.scatter(y_true, prediction, alpha=0.7, label="Phase 2 LSTM")
    ax.plot([0, 150], [0, 150], "k--", linewidth=1)
    ax.set(
        xlim=(0, 150),
        ylim=(0, 150),
        xlabel="True RUL (cycles)",
        ylabel="Predicted RUL (cycles)",
        title="Official FD001 test set: last observed cycle",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_predictions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(xgb_prediction - y_true, bins=20, alpha=0.6, label="XGBoost")
    ax.hist(prediction - y_true, bins=20, alpha=0.6, label="LSTM")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Prediction error (predicted - true RUL)", ylabel="Engines")
    ax.set_title("Residual distributions on official test engines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_residuals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        test_predictions["unit"],
        np.abs(xgb_prediction - y_true),
        marker=".",
        label="XGBoost",
    )
    ax.plot(
        test_predictions["unit"],
        np.abs(prediction - y_true),
        marker=".",
        label="LSTM",
    )
    ax.set(
        xlabel="Engine ID",
        ylabel="Absolute error (cycles)",
        title="Per-engine test error",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "test_engine_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = stage_results.pivot(index="stage", columns="model", values="RMSE")[
        ["XGBoost", "LSTM"]
    ]
    pivot.plot(kind="bar", ax=ax, color=["#555555", "#2878b5"])
    ax.set(ylabel="RMSE (cycles)", xlabel="", title="Last-cycle test error by true-RUL stage")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figure_dir / "test_stage_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        comparison["model"],
        comparison["RMSE"].astype(float),
        color=["#999999" if name != "LSTM" else "#2878b5" for name in comparison["model"]],
    )
    ax.set(ylabel="RMSE (cycles)", title="Model comparison on the same 100 test engines")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [entry["epoch"] for entry in result.history],
        [entry["train_loss"] for entry in result.history],
    )
    ax.set(xlabel="Epoch", ylabel="Huber loss", title="Final LSTM fit on all 100 training engines")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_training_loss.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics["metrics_true_RUL"], indent=2))
    print(f"Saved final model to {checkpoint_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
