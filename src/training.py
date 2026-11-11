"""Reproducible LSTM training, checkpointing, and batched inference."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .lstm import LSTMRegressor
from .sequence_data import SequenceBatch


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    min_delta: float = 1e-4
    optimizer: str = "adamw"
    loss: str = "huber"
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_threads: int = 1

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if self.loss not in {"huber", "mse"}:
            raise ValueError("loss must be 'huber' or 'mse'")
        if self.scheduler_patience < 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError("invalid learning-rate scheduler settings")
        if self.max_grad_norm <= 0 or self.num_threads < 1:
            raise ValueError("max_grad_norm and num_threads must be positive")


@dataclass
class TrainingResult:
    model: LSTMRegressor
    history: list
    best_epoch: int
    best_validation_loss: float
    device: str
    checkpoint_path: Path
    history_path: Path


def _resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported training device: {device.type}")
    return device


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_loader(batch, batch_size, shuffle, seed):
    if batch.x.ndim != 3 or batch.x.shape[0] != len(batch.y):
        raise ValueError("batch must contain aligned (n, time, features) windows and targets")
    if len(batch.y) == 0:
        raise ValueError("training and validation batches must not be empty")
    if not np.isfinite(batch.x).all() or not np.isfinite(batch.y).all():
        raise ValueError("training windows and targets must be finite")
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(batch.x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(batch.y, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def _run_epoch(model, loader, criterion, device, optimizer=None, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_samples = 0
    for sequences, targets in loader:
        sequences, targets = sequences.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        total_loss += loss.item() * len(targets)
        n_samples += len(targets)
    return total_loss / n_samples


def train_lstm(
    train_batch,
    validation_batch,
    checkpoint_path,
    model_config=None,
    config=None,
):
    """Train and restore the best checkpoint.

    With validation data, early stopping and checkpoint selection use only those
    engine-disjoint windows. Without validation data, train for exactly ``epochs``;
    this supports final refitting after selecting the epoch count on validation.
    """
    config = TrainingConfig() if config is None else config
    model_config = {"input_size": train_batch.x.shape[2]} if model_config is None else dict(model_config)
    if validation_batch is not None and set(train_batch.unit).intersection(
        validation_batch.unit
    ):
        raise ValueError("training and validation batches must use disjoint engines")
    if train_batch.x.shape[2] != model_config.get("input_size"):
        raise ValueError("model input_size must match the sequence feature count")

    _seed_everything(config.seed)
    torch.set_num_threads(config.num_threads)
    device = _resolve_device(config.device)
    model = LSTMRegressor(**model_config).to(device)
    train_loader = _make_loader(train_batch, config.batch_size, True, config.seed)
    validation_loader = (
        None
        if validation_batch is None
        else _make_loader(validation_batch, config.batch_size, False, config.seed)
    )

    criterion = nn.SmoothL1Loss() if config.loss == "huber" else nn.MSELoss()
    optimizer_type = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_type(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.history.json"
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, criterion, device, optimizer, config.max_grad_norm
        )
        validation_loss = (
            None
            if validation_loader is None
            else _run_epoch(model, validation_loader, criterion, device)
        )
        if validation_loss is not None:
            scheduler.step(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
        )

        if validation_loss is None or validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "model_config": model.get_config(),
                "training_config": asdict(config),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    history_path.write_text(
        json.dumps(
            {
                "model_config": model_config,
                "training_config": asdict(config),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        device=str(device),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def predict_lstm(model, batch, batch_size=512, device=None):
    """Return one prediction per supplied window in the original window order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch.x.ndim != 3:
        raise ValueError("batch.x must have shape (n, time, features)")
    if len(batch.x) == 0:
        return np.empty(0, dtype=np.float32)
    model_device = next(model.parameters()).device if device is None else torch.device(device)
    loader = _make_loader(batch, batch_size, False, seed=0)
    model = model.to(model_device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in loader:
            predictions.append(model(sequences.to(model_device)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)
