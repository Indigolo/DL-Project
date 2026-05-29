"""Inference entrypoint for the trained LSTM baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data import SequencePreprocessor, autoregressive_predict, load_input_frame
from src.model import ForecastModel


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """Load the rows that need predictions."""
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index)
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def load_input_frame_or_train(input_dir: Path) -> pd.DataFrame:
    """Load the benchmark input frame, or train.csv as a local fallback."""
    candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
        input_dir / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def load_history_frame(input_dir: Path) -> pd.DataFrame:
    """Load observed target history for local validation or private inference."""
    candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
        input_dir / "train.csv",
        input_dir.parent / "train.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "target" in frame.columns and frame["target"].notna().any():
            return frame.loc[frame["target"].notna()].copy()
    searched = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        "Could not find observed history with a non-null `target` column. "
        f"Searched: {searched}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint must contain `state_dict`, `model_config`, and `preprocessor`.")

    preprocessor = SequencePreprocessor.from_checkpoint(checkpoint["preprocessor"])
    model = ForecastModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    forecast_index = load_forecast_index(args.input_dir)
    future_covariates = load_input_frame(args.input_dir)
    history_frame = load_history_frame(args.input_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = autoregressive_predict(
        model=model,
        history_frame=history_frame,
        forecast_index=forecast_index,
        preprocessor=preprocessor,
        future_covariates=future_covariates,
        device=device,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
