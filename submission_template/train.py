"""Train a PatchTST baseline for the assignment dataset."""

from __future__ import annotations

import argparse
import random
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


def train_epoch(
    model: ForecastModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-steps", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_frame = pd.read_csv(args.train)
    preprocessor = SequencePreprocessor.fit(train_frame, context_length=args.context_length)
    prepared = preprocessor.fit_transform_train(train_frame)
    train_cutoffs = split_cutoffs(prepared, validation_steps=args.validation_steps)
    dataset = WindowDataset(prepared, preprocessor, train_cutoffs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ForecastModel(
        input_size=preprocessor.input_size,
        context_length=args.context_length,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        series_count=preprocessor.series_count,
        series_embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_rmse = float("inf")
    best_payload: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        metrics = evaluate_rollout(model, prepared, preprocessor, train_cutoffs, device)
        print(
            f"epoch={epoch:02d} "
            f"train_huber={train_loss:.5f} "
            f"val_mae={metrics['mae']:.5f} "
            f"val_rmse={metrics['rmse']:.5f}"
        )

        if best_payload is None or not np.isfinite(best_rmse) or metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_payload = {
                "state_dict": model.state_dict(),
                "model_config": {
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
                },
                "preprocessor": preprocessor.to_checkpoint(),
                "metrics": metrics,
            }

    if best_payload is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, args.checkpoint)
    print(f"saved_checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
