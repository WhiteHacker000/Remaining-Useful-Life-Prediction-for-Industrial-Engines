# Phase 2 experiment results

Every deep-learning experiment of Phase 2 is saved here, so notebooks 07–10 can show the results without retraining. To retrain an experiment from scratch, set `RETRAIN = True` in the notebook that runs it. Results are then recomputed and these files are overwritten.

All cross-validation (CV) runs use the same 5 folds over engine ids as Phase 1 (`src/models.py: engine_folds`, seed 42). Inside each fold, 10% of the training engines are held out for early stopping. CV errors are RMSE in cycles against the capped target (RUL capped at 125), over every cycle of the held-out engines.

## File pairs

Most experiments are stored as two files with the same name:

| File | Content |
|---|---|
| `<name>.json` | settings and summary of the run |
| `<name>_pred.csv` | one prediction per row (engine `unit`, `cycle`, `true_rul`, prediction) |

**CV runs** (`cv_*`, `abl_*`, `seed_*`), JSON fields:

| Field | Meaning |
|---|---|
| `model` | `lstm`, `tcn`, `transformer`, `features` (feature branch alone) or `hybrid-<encoder>` |
| `config` | training settings (`src/training.py: TrainConfig`): window length, epochs, dropout, weight decay, early stopping, input noise, padding mask |
| `seed` | random seed of the run |
| `fold_rmse`, `cv_rmse`, `cv_std` | RMSE of each fold, their mean and standard deviation |
| `best_epochs` | epoch with the lowest inner-validation error, per fold |
| `history` | training and validation RMSE per epoch and fold (used for the learning-curve plots) |
| `minutes` | training time |

The `_pred.csv` file holds the out-of-fold prediction for every cycle of every training engine (`pred`).

**Final test runs** (`test_*`): trained on all 100 training engines with seeds 0, 1 and 2. The JSON lists the best epoch and validation RMSE per seed. The `_pred.csv` file has one column per seed (`pred_seed0`, `pred_seed1`, `pred_seed2`) and their average (`pred`), for every test cycle. The trained weights are in `outputs/models/`.

## Experiments by notebook

| Notebook | Files | What they hold |
|---|---|---|
| 07 Sequence models | `cv_lstm_w30`, `cv_tcn_w30`, `cv_transformer_w30`, `cv_sequence_models.csv` | the three architectures on 30-cycle windows, and the comparison table with Phase 1's XGBoost |
| 08 Hybrid model | `cv_features`, `cv_hybrid_lstm_w30`, `cv_hybrid_tcn_w30`, `cv_hybrid_transformer_w30`, `cv_hybrid_comparison.csv`, `cv_weak_spots.csv` | the feature branch alone, the hybrid with each encoder, the comparison with late fusion, and the Phase 1 weak-spot analysis |
| 09 Ablation studies | `abl_*` | one change at a time to the LSTM hybrid (window 50/80, no dropout, no weight decay, no early stopping, input noise, no padding mask) and the LSTM alone with windows 50/80 |
| | `abl_hybrid_lstm_input_noise_first_run` | first run of the input-noise ablation; the rerun with an identical configuration (`abl_hybrid_lstm___input_noise______0_05_`) gave a different result, which showed that training is not bit-reproducible |
| | `cv_hybrid_lstm_60ep_noise` | the combination of 60 epochs and input noise |
| | `seed_hybrid_lstm_default_s1/s2`, `seed_hybrid_lstm_60ep_s1/s2` | the default and the final configuration retrained with seeds 1 and 2 |
| | `ablation_summary.csv`, `final_configuration.csv`, `seed_variability.csv` | the summary tables shown in the notebook |
| 10 Test evaluation | `test_lstm_w30`, `test_tcn_w30`, `test_transformer_w30`, `test_hybrid_lstm_w30_60ep` | final models on the official test set (3 seeds each) |
| | `test_benchmark_dl.csv`, `test_significance.csv`, `test_error_by_stage_dl.csv`, `phase1_vs_phase2.csv` | last-cycle benchmark with bootstrap CIs, paired Wilcoxon tests, error by degradation stage, and the Phase 1 vs Phase 2 summary |

Phase 1's XGBoost predictions for every test cycle, used in notebook 10, are in `outputs/results/xgb_test_predictions.csv` (created by `scripts/xgb_test_predictions.py`).

## Reading the ablations

Retraining the same configuration with different seeds changes the CV RMSE by about ±0.5–0.7 cycles (`seed_variability.csv`). Ablation effects smaller than that cannot be separated from training noise. The effects that are clearly larger are removing the feature branch (+4.46) and giving the LSTM alone a longer window (14.01 → 11.87).
