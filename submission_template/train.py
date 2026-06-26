"""Train a PatchTST baseline for the assignment dataset."""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    SequencePreprocessor,
    WindowDataset,
    evaluate_rollout,
    split_cutoffs,
)
from src.model import ForecastModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(max(seconds, 0.0))))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay, stepped once per training batch."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: ForecastModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    loss_fn = torch.nn.HuberLoss()
    for x_batch, series_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        series_batch = series_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(x_batch, series_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an autoregressive PatchTST baseline.")
    parser.add_argument("--train", required=True, type=Path, help="Path to train.csv")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Where to store the checkpoint")
    parser.add_argument("--context-length", type=int, default=168, help="History window used by the model")
    parser.add_argument("--patch-len", type=int, default=24, help="Length of each patch (in time steps)")
    parser.add_argument("--stride", type=int, default=12, help="Stride between consecutive patches")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=128, help="Transformer feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Linear LR warmup steps before cosine decay kicks in",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Cosine decay floor as a fraction of --learning-rate (0 decays to 0)",
    )
    parser.add_argument("--validation-steps", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing checkpoint to continue training from. Architecture and "
        "preprocessing come from the checkpoint, not the CLI flags above.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop early after this many epochs without validation RMSE "
        "improvement. Set to <=0 to disable early stopping.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_frame = pd.read_csv(args.train)

    resume_checkpoint: dict[str, object] | None = None
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Missing checkpoint to resume from: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        preprocessor = SequencePreprocessor.from_checkpoint(resume_checkpoint["preprocessor"])
        model_config = resume_checkpoint["model_config"]
    else:
        preprocessor = SequencePreprocessor.fit(train_frame, context_length=args.context_length)
        model_config = {
            "input_size": preprocessor.input_size,
            "context_length": args.context_length,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "series_count": preprocessor.series_count,
            "series_embedding_dim": args.embedding_dim,
        }

    prepared = preprocessor.fit_transform_train(train_frame)
    train_cutoffs = split_cutoffs(prepared, validation_steps=args.validation_steps)
    dataset = WindowDataset(prepared, preprocessor, train_cutoffs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ForecastModel(**model_config)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["state_dict"])
        print(f"resumed_from={args.resume} prior_val_rmse={resume_checkpoint['metrics']['rmse']:.5f}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    best_payload: dict[str, object] | None
    if resume_checkpoint is not None:
        best_rmse = float(resume_checkpoint["metrics"]["rmse"])
        best_payload = resume_checkpoint
    else:
        best_rmse = float("inf")
        best_payload = None
    epochs_without_improvement = 0
    training_start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        train_loss = train_epoch(model, loader, optimizer, scheduler, device)
        metrics = evaluate_rollout(model, prepared, preprocessor, train_cutoffs, device)
        epoch_duration = time.monotonic() - epoch_start
        elapsed = time.monotonic() - training_start
        avg_epoch_duration = elapsed / epoch
        eta_seconds = avg_epoch_duration * (args.epochs - epoch)
        print(
            f"epoch={epoch:02d} "
            f"train_huber={train_loss:.5f} "
            f"val_mae={metrics['mae']:.5f} "
            f"val_rmse={metrics['rmse']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"epoch_time={_format_duration(epoch_duration)} "
            f"elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta_seconds)}"
        )

        if best_payload is None or not np.isfinite(best_rmse) or metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            epochs_without_improvement = 0
            best_payload = {
                "state_dict": model.state_dict(),
                "model_config": model_config,
                "preprocessor": preprocessor.to_checkpoint(),
                "metrics": metrics,
            }
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"early_stopping=triggered epoch={epoch:02d} "
                    f"epochs_without_improvement={epochs_without_improvement} "
                    f"best_val_rmse={best_rmse:.5f}"
                )
                break

    if best_payload is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, args.checkpoint)
    print(f"saved_checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
