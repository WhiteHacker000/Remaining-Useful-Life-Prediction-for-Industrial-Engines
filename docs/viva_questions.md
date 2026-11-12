# Viva Preparation — Aircraft Engine RUL Prediction

Answers below describe the implemented FD001 experiments and distinguish validation evidence from official-test results.

## Why is remaining useful life capped at 125?

Early-life sensors contain relatively little visible degradation, so a linear RUL target asks the model to distinguish large remaining-life values from weak signals. The piecewise-linear target is `y = min(RUL, 125)`. Phase 1 measured lower validation error with a cap than with a linear target. The cap is a modeling target, not a claim that engines cannot operate longer than 125 cycles. In the official test set, 11 final cutoffs have true RUL greater than 125; metrics against uncapped truth therefore include the intentional early-life target mismatch.

## Why split by engine ID rather than by individual rows?

Adjacent rows from one engine are highly correlated and overlapping sequence windows share most of their observations. A random row split can place nearly duplicate histories in both training and validation, making performance look unrealistically good. Phase 2 assigns whole engine IDs to fit or validation before it creates any windows; the official test engines remain separate.

## How was Phase 2 validation designed?

The 100 run-to-failure training engines were split 80/20 by engine ID with seed 42. For each of the 20 held-out engines, a deterministic pre-failure cutoff was chosen using seed 43, with true RUL from 1 through 125 when possible. This simulates a censored observation instead of always validating at the failure cycle, where every label would be zero. The same cutoff rows are used to compare validation models.

## Why must normalization be fitted only on training engines?

Fitting a scaler on validation or test rows lets information about those engine populations affect the learned transformation. This is leakage, even without labels. `EngineStandardizer` is fitted on the 80 fit engines during selection; its learned mean and scale are then applied unchanged to validation. For the final refit, it is fitted on the 100 training engines and applied to the official test data.

## How are the LSTM sequences and labels constructed?

Within each engine, rows are sorted by cycle. A window ends at cycle `t` and its target is the RUL at `t`; no window can cross an engine boundary. Short histories are left-padded by repeating the first normalized sensor vector. At final test inference, only the window ending at each engine's last observed cycle is predicted.

## What model was used?

One LSTM layer receives 14 standardized sensor channels over 50 cycles and uses hidden size 32. The final recurrent output passes through a dropout and fully connected regression head, producing one scalar RUL estimate. The model was trained using AdamW and Huber loss, with gradient clipping and learning-rate scheduling.

## What is early stopping, and how was it used?

Early stopping monitors a validation loss that is computed on engines disjoint from training. If it does not improve by the configured minimum delta for the patience interval, training stops and the best checkpoint is restored. The ablation runner uses this during model selection. After selecting the model and epoch count, the final model is refit on all 100 training engines for 32 epochs, with no validation or test examples in that fit.

## What did the ablation show?

On the single validation holdout, window length 50 gave RMSE 8.359 and MAE 6.521; lengths 20 and 30 had RMSE 18.385 and 12.895. Dropout and weight-decay settings were also varied at window length 30. The results selected the 50-cycle configuration based only on validation RMSE. The large gap to official-test RMSE is a limitation and should not be hidden.

## Did the LSTM outperform XGBoost?

No. On the same 100 official test engines and uncapped true RUL, XGBoost achieved RMSE 10.754, MAE 8.085, and NASA score 161.109. The selected LSTM achieved 14.857, 10.559, and 309.965. A lower metric is better. The LSTM's validation RMSE of 8.359 did not generalize. The result is reported as measured; the test set was not used for further tuning.

## Why can a sequence model perform worse than engineered-feature XGBoost?

An LSTM can learn temporal patterns, but it also has more parameters and training variance. A single 80-engine fit split and one holdout may not represent the official test distribution well. C-MAPSS FD001 is small; engineered cycle, baseline-deviation, slope, and health-index features encode strong domain structure that a data-hungry model may not learn reliably. The evidence supports the engineered XGBoost model for this experiment, not a general claim that LSTMs are inferior.

## What does the NASA score measure?

The score is asymmetric: it penalizes predictions that overestimate RUL more heavily than predictions that underestimate it. For error `d = predicted - true`, the project uses `exp(-d/13)-1` when `d < 0` and `exp(d/10)-1` otherwise, summed across evaluated cases. Lower is better. This reflects the safety concern of predicting too much remaining life, but it should be interpreted alongside RMSE and MAE.

## Why evaluate only the last observed test cycle?

The official C-MAPSS test labels give the remaining life at each engine's final observed cycle. Evaluating one prediction per engine at that cutoff matches the benchmark protocol and ensures a fair comparison with Phase 1. Earlier test cycles are not independent test cases and do not have official labels in the supplied test file; reconstructed labels are useful for stage analysis but are not the primary final metric.

## Why was a hybrid model not implemented?

The standalone LSTM did not improve on the official test set, and the official set has already been used for the prespecified final comparison. Developing and selecting a hybrid after seeing those test metrics and then evaluating it on the same engines would make the test set part of model selection. A defensible hybrid study needs new engine-level validation and a new untouched final evaluation population.

## What are the main limitations and future directions?

The study is limited to one simulated FD001 operating condition and fault mode, one validation split, a single LSTM configuration selected from a compact sweep, and 100 official test engines. Generalization to FD002–FD004, real engines, and other seeds is unverified. Next work should use repeated engine-level validation, pre-register the selection protocol, and evaluate once on an untouched population. The Phase 2 final-cycle stage groups contain only 100 engine observations and should not be conflated with Phase 1 stage analysis over all 13,096 test cycles.
