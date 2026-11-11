"""Engine-level Phase 2 validation experiments and configuration selection."""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RUL_CAP, SEED
from .evaluation import mae, rmse
from .features import FeatureBuilder
from .models import make_model
from .sequence_data import EngineStandardizer, make_sequence_windows, split_engine_ids
from .training import TrainingConfig, predict_lstm, train_lstm


def phase2_experiment_specs():
    """Return a compact sweep over context length and regularization strength."""
    return [
        {"name": "window_20_base", "window_size": 20, "dropout": 0.2, "weight_decay": 1e-4},
        {"name": "window_30_base", "window_size": 30, "dropout": 0.2, "weight_decay": 1e-4},
        {"name": "window_50_base", "window_size": 50, "dropout": 0.2, "weight_decay": 1e-4},
        {"name": "window_30_dropout_0", "window_size": 30, "dropout": 0.0, "weight_decay": 1e-4},
        {"name": "window_30_dropout_04", "window_size": 30, "dropout": 0.4, "weight_decay": 1e-4},
        {"name": "window_30_no_weight_decay", "window_size": 30, "dropout": 0.2, "weight_decay": 0.0},
    ]


def select_best_experiment(results):
    """Select solely by validation RMSE against the capped target."""
    if not results:
        raise ValueError("at least one validation result is required")
    return min(results, key=lambda row: (row["validation_rmse_capped"], row["name"]))


def validation_cutoff_cycles(frame, seed=SEED):
    """Choose deterministic pre-failure cutoffs with capped-RUL coverage."""
    rng = np.random.default_rng(seed)
    cutoffs = {}
    for unit, engine in frame.groupby("unit", sort=True):
        final_cycle = int(engine["cycle"].max())
        max_rul = min(RUL_CAP, final_cycle - 1)
        if max_rul < 1:
            raise ValueError(f"engine {unit} has no pre-failure validation cycle")
        remaining_life = int(rng.integers(1, max_rul + 1))
        cutoffs[int(unit)] = final_cycle - remaining_life
    return cutoffs


def _metric_pair(y_true, prediction):
    return {"RMSE": rmse(y_true, prediction), "MAE": mae(y_true, prediction)}


def run_phase2_validation(
    train,
    output_dir,
    best_params_path,
    seed=SEED,
    epochs=40,
    batch_size=256,
    device="cpu",
    experiment_specs=None,
):
    """Tune on a seeded engine holdout without reading or evaluating the test set."""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_ids, validation_ids = split_engine_ids(
        train["unit"].unique(), validation_fraction=0.2, seed=seed
    )
    train_frame = train[train["unit"].isin(training_ids)].copy()
    validation_full = train[train["unit"].isin(validation_ids)].copy()
    cutoff_cycles = validation_cutoff_cycles(validation_full, seed=seed + 1)
    cutoff_for_row = validation_full["unit"].map(cutoff_cycles)
    validation_frame = validation_full[
        validation_full["cycle"] <= cutoff_for_row
    ].copy()

    standardizer = EngineStandardizer().fit(train_frame, training_ids)
    normalized_train = standardizer.transform(train_frame)
    normalized_validation = standardizer.transform(validation_frame)

    feature_builder = FeatureBuilder().fit(train_frame)
    train_features = feature_builder.transform(train_frame)
    validation_features = feature_builder.transform(validation_frame)
    train_target = np.minimum(
        train_frame.loc[train_features.index, "RUL"].to_numpy(), RUL_CAP
    )
    best_params = json.loads(Path(best_params_path).read_text())["xgboost"]
    xgb_model = make_model("xgboost", **best_params).fit(
        train_features[feature_builder.feature_names], train_target
    )
    last_indices = validation_frame.groupby("unit")["cycle"].idxmax().to_numpy()
    xgb_prediction = np.clip(
        xgb_model.predict(
            validation_features.loc[last_indices, feature_builder.feature_names]
        ),
        0,
        None,
    )
    y_validation = validation_frame.loc[last_indices, "RUL"].to_numpy()
    y_validation_capped = np.minimum(y_validation, RUL_CAP)
    xgb_validation_metrics = {
        **_metric_pair(y_validation_capped, xgb_prediction),
        "RMSE_true_RUL": rmse(y_validation, xgb_prediction),
        "MAE_true_RUL": mae(y_validation, xgb_prediction),
    }

    specs = phase2_experiment_specs() if experiment_specs is None else experiment_specs
    results = []
    predictions = [
        pd.DataFrame(
            {
                "model": "XGBoost",
                "experiment": "phase1_tuned",
                "unit": validation_frame.loc[last_indices, "unit"].to_numpy(),
                "true_rul": y_validation,
                "capped_rul": y_validation_capped,
                "prediction": xgb_prediction,
            }
        )
    ]
    histories = {}

    for spec in specs:
        name = spec["name"]
        validation_windows = make_sequence_windows(
            normalized_validation,
            window_size=spec["window_size"],
            target_cap=RUL_CAP,
            last_only=True,
        )
        model_config = {
            "input_size": len(standardizer.sensor_columns),
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": spec["dropout"],
        }
        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=spec["weight_decay"],
            seed=seed,
            device=device,
        )
        train_windows = make_sequence_windows(
            normalized_train,
            window_size=spec["window_size"],
            target_cap=RUL_CAP,
        )
        result = train_lstm(
            train_windows,
            validation_windows,
            checkpoint_dir / f"{name}.pt",
            model_config=model_config,
            config=training_config,
        )
        prediction = predict_lstm(
            result.model, validation_windows, batch_size=batch_size, device=result.device
        )
        row = {
            **spec,
            "validation_rmse_capped": rmse(validation_windows.y, prediction),
            "validation_mae_capped": mae(validation_windows.y, prediction),
            "validation_rmse_true_RUL": rmse(y_validation, prediction),
            "validation_mae_true_RUL": mae(y_validation, prediction),
            "best_epoch": result.best_epoch,
            "best_validation_huber_loss": result.best_validation_loss,
            "checkpoint": str(result.checkpoint_path.relative_to(output_dir)),
            "history": str(result.history_path.relative_to(output_dir)),
            "model_config": model_config,
            "training_config": asdict(training_config),
        }
        results.append(row)
        predictions.append(
            pd.DataFrame(
                {
                    "model": "LSTM",
                    "experiment": name,
                    "unit": validation_windows.unit,
                    "true_rul": y_validation,
                    "capped_rul": validation_windows.y,
                    "prediction": prediction,
                }
            )
        )
        histories[name] = result.history
        print(
            f"{name}: validation RMSE={row['validation_rmse_capped']:.3f}, "
            f"best epoch={result.best_epoch}"
        )

    selected_result = select_best_experiment(results)
    selected = {
        "name": selected_result["name"],
        "window_size": selected_result["window_size"],
        "model_config": selected_result["model_config"],
        "training_config": selected_result["training_config"],
        "best_epoch": selected_result["best_epoch"],
        "validation_rmse_capped": selected_result["validation_rmse_capped"],
        "validation_mae_capped": selected_result["validation_mae_capped"],
    }
    summary = {
        "dataset": "NASA C-MAPSS FD001",
        "seed": seed,
        "split_method": "80/20 split of training engine IDs before window construction",
        "training_engine_ids": training_ids.tolist(),
        "validation_engine_ids": validation_ids.tolist(),
        "validation_cutoff_cycles": cutoff_cycles,
        "validation_metric_for_selection": "RMSE against RUL capped at 125 at one deterministic pre-failure cutoff per engine",
        "sequence_padding": "Repeat the first observed normalized sensor vector on the left.",
        "phase1_xgboost_validation": {
            "parameters": best_params,
            "metrics": xgb_validation_metrics,
        },
        "selected_experiment": selected,
        "experiments": results,
    }
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(results).to_csv(output_dir / "validation_experiments.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )

    figure_dir = output_dir.parent.parent / "figures" / "phase2"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["XGBoost"] + [row["name"].replace("window_", "w") for row in results]
    values = [xgb_validation_metrics["RMSE"]] + [
        row["validation_rmse_capped"] for row in results
    ]
    ax.bar(labels, values, color=["#555555"] + ["#2878b5"] * len(results))
    ax.set_ylabel("Validation RMSE (RUL capped at 125)")
    ax.set_title("Engine-holdout model and LSTM ablation comparison")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_ablation.png", dpi=160)
    plt.close(fig)

    selected_history = histories[selected["name"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [entry["epoch"] for entry in selected_history],
        [entry["train_loss"] for entry in selected_history],
        label="Training",
    )
    ax.plot(
        [entry["epoch"] for entry in selected_history],
        [entry["validation_loss"] for entry in selected_history],
        label="Validation",
    )
    ax.set(xlabel="Epoch", ylabel="Huber loss", title="Selected LSTM training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_history.png", dpi=160)
    plt.close(fig)
    return summary
