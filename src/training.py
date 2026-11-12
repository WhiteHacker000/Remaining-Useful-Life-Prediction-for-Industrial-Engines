"""Training, early stopping, engine-level cross-validation and caching for the DL models."""

import copy
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from .config import N_FOLDS, RESULTS_DIR, RUL_CAP, SEED
from .dl_models import make_dl_model
from .evaluation import rmse
from .models import engine_folds
from .sequences import SequenceBuilder, capped

DL_DIR = RESULTS_DIR / "dl"
torch.set_num_threads(max(1, torch.get_num_threads()))


@dataclass
class TrainConfig:
    window: int = 30
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4      # L2 penalty (AdamW-style decoupled)
    dropout: float = 0.2
    patience: int = 8               # early stopping: epochs without validation improvement
    grad_clip: float = 1.0          # max gradient norm
    noise_std: float = 0.0          # Gaussian input noise (data augmentation), in z units
    use_mask: bool = True           # give the model the padding-mask channel
    early_stopping: bool = True
    device: str = "auto"            # "auto": Apple GPU (MPS) for convolutional models, else CPU


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(model, cfg):
    """Convolutions run much faster on the Apple GPU; small LSTMs/Transformers are faster on CPU."""
    if cfg.device != "auto":
        return torch.device(cfg.device)
    has_conv = any(isinstance(m, torch.nn.Conv1d) for m in model.modules())
    return torch.device("mps" if has_conv and torch.backends.mps.is_available() else "cpu")


def _tensors(data, cfg):
    x = data["x"] if cfg.use_mask else data["x"][:, :, :-1]
    t = {"x": torch.from_numpy(np.ascontiguousarray(x))}
    if "f" in data:
        t["f"] = torch.from_numpy(data["f"])
    return t


def predict(model, data, cfg, batch_size=2048):
    """Predicted RUL in cycles, clipped at 0."""
    model.eval()
    dev = next(model.parameters()).device
    t = _tensors(data, cfg)
    out = []
    with torch.no_grad():
        for i in range(0, len(t["x"]), batch_size):
            xb = t["x"][i:i + batch_size].to(dev)
            fb = t["f"][i:i + batch_size].to(dev) if "f" in t else None
            out.append(model(xb, fb).cpu().numpy())
    return np.clip(np.concatenate(out) * RUL_CAP, 0, None)


def fit(model, train, val, cfg, seed=SEED, verbose=False):
    """Train with Adam, MSE on the capped target scaled to [0, 1].

    Regularisation: dropout (inside the model), weight decay, gradient clipping,
    learning-rate halving on plateau, early stopping on the validation engines
    (the best weights are restored at the end), optional input noise.
    """
    set_seed(seed)
    dev = pick_device(model, cfg)
    model.to(dev)
    t = {k: v.to(dev) for k, v in _tensors(train, cfg).items()}
    y = torch.from_numpy(capped(train["y"]) / RUL_CAP).to(dev)
    y_val = capped(val["y"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = torch.nn.MSELoss()
    n_sensor = t["x"].shape[2] - (1 if cfg.use_mask else 0)

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(y), device=dev)
        total = 0.0
        for i in range(0, len(y), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = t["x"][idx]
            if cfg.noise_std > 0:
                noise = torch.randn_like(xb[:, :, :n_sensor]) * cfg.noise_std
                xb = torch.cat([xb[:, :, :n_sensor] + noise, xb[:, :, n_sensor:]], dim=2)
            fb = t["f"][idx] if "f" in t else None
            loss = loss_fn(model(xb, fb), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(idx)

        val_rmse = rmse(y_val, predict(model, val, cfg))
        train_rmse = float(np.sqrt(total / len(y))) * RUL_CAP
        history.append({"epoch": epoch + 1, "train_rmse": train_rmse, "val_rmse": val_rmse,
                        "lr": opt.param_groups[0]["lr"]})
        sched.step(val_rmse)
        if verbose:
            print(f"epoch {epoch + 1:3d}  train {train_rmse:6.2f}  val {val_rmse:6.2f}")
        if val_rmse < best - 1e-3:
            # deepcopy: state_dict() returns references that keep changing during training
            best, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if cfg.early_stopping and bad >= cfg.patience:
                break
    if cfg.early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), pd.DataFrame(history)


def _build(train_df, eval_df, cfg, with_features, seed):
    """Split train engines into fit/inner-validation, fit the builder on fit engines only."""
    units = np.array(sorted(train_df["unit"].unique()))
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=max(2, int(round(0.1 * len(units)))), replace=False)
    fit_df = train_df[~train_df["unit"].isin(val_units)]
    val_df = train_df[train_df["unit"].isin(val_units)]
    sb = SequenceBuilder(cfg.window, with_features=with_features).fit(fit_df)
    return sb.transform(fit_df), sb.transform(val_df), sb.transform(eval_df), sb


def _cache_paths(key):
    DL_DIR.mkdir(parents=True, exist_ok=True)
    return DL_DIR / f"{key}.json", DL_DIR / f"{key}_pred.csv"


def run_cv(key, name, train_df, cfg, seed=SEED, n_folds=N_FOLDS, retrain=False):
    """5-fold CV over engine ids (same folds as Phase 1). Returns (summary dict, oof DataFrame).

    Inside each fold: 10% of the training engines are held out for early stopping,
    the model never sees the evaluation engines. Results are cached under outputs/results/dl.
    """
    meta_path, pred_path = _cache_paths(key)
    if meta_path.exists() and pred_path.exists() and not retrain:
        return json.loads(meta_path.read_text()), pd.read_csv(pred_path)

    with_features = name == "features" or name.startswith("hybrid")
    start, scores, oof, histories, epochs = time.time(), [], [], [], []
    for k, val_units in enumerate(engine_folds(train_df["unit"], n_folds, SEED)):
        is_eval = train_df["unit"].isin(val_units)
        tr, va, ev, sb = _build(train_df[~is_eval], train_df[is_eval], cfg, with_features, seed + k)
        model = make_dl_model(name, tr["x"].shape[2] - (0 if cfg.use_mask else 1), cfg.window,
                              n_feat=tr["f"].shape[1] if with_features else None, dropout=cfg.dropout)
        model, hist = fit(model, tr, va, cfg, seed=seed + k)
        pred = predict(model, ev, cfg)
        scores.append(rmse(capped(ev["y"]), pred))
        epochs.append(int(hist["val_rmse"].idxmin()) + 1)
        histories.append(hist.assign(fold=k + 1))
        oof.append(pd.DataFrame({"unit": ev["unit"], "cycle": ev["cycle"], "true_rul": ev["y"], "pred": pred}))

    summary = {"key": key, "model": name, "config": asdict(cfg), "seed": seed,
               "fold_rmse": [float(s) for s in scores], "cv_rmse": float(np.mean(scores)),
               "cv_std": float(np.std(scores)), "best_epochs": epochs,
               "minutes": round((time.time() - start) / 60, 2),
               "history": pd.concat(histories).to_dict(orient="list")}
    oof = pd.concat(oof, ignore_index=True)
    meta_path.write_text(json.dumps(summary))
    oof.to_csv(pred_path, index=False)
    return summary, oof


def run_final(key, name, train_df, test_df, cfg, seeds=(0, 1, 2), retrain=False):
    """Train on all 100 training engines (10% held out for early stopping) with several
    seeds and predict every test cycle. Returns a DataFrame with one prediction column per seed."""
    meta_path, pred_path = _cache_paths(key)
    if meta_path.exists() and pred_path.exists() and not retrain:
        return json.loads(meta_path.read_text()), pd.read_csv(pred_path)

    with_features = name == "features" or name.startswith("hybrid")
    preds, info, start = None, [], time.time()
    model_dir = RESULTS_DIR.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        tr, va, te, sb = _build(train_df, test_df, cfg, with_features, SEED + s)
        model = make_dl_model(name, tr["x"].shape[2] - (0 if cfg.use_mask else 1), cfg.window,
                              n_feat=tr["f"].shape[1] if with_features else None, dropout=cfg.dropout)
        model, hist = fit(model, tr, va, cfg, seed=SEED + s)
        torch.save(model.state_dict(), model_dir / f"{key}_seed{s}.pt")
        if preds is None:
            preds = pd.DataFrame({"unit": te["unit"], "cycle": te["cycle"], "true_rul": te["y"]})
        preds[f"pred_seed{s}"] = predict(model, te, cfg)
        info.append({"seed": s, "best_epoch": int(hist["val_rmse"].idxmin()) + 1,
                     "val_rmse": float(hist["val_rmse"].min())})
    preds["pred"] = preds[[c for c in preds if c.startswith("pred_seed")]].mean(axis=1)
    summary = {"key": key, "model": name, "config": asdict(cfg), "seeds": info,
               "minutes": round((time.time() - start) / 60, 2)}
    meta_path.write_text(json.dumps(summary))
    preds.to_csv(pred_path, index=False)
    return summary, preds
