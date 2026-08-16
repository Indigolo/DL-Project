"""Preprocessing, windowing, and autoregressive inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SERIES_COL = "series_id"
TIME_COL = "timestamp"
TARGET_COL = "target"


def _safe_std(values: pd.Series) -> float:
    std = float(values.std())
    return std if np.isfinite(std) and std > 1e-8 else 1.0


def _base_time_features(timestamps: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(timestamps)
    hour = dt.dt.hour.to_numpy()
    dayofweek = dt.dt.dayofweek.to_numpy()
    dayofyear = dt.dt.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dayofweek / 7.0),
            "dow_cos": np.cos(2 * np.pi * dayofweek / 7.0),
            "doy_sin": np.sin(2 * np.pi * dayofyear / 366.0),
            "doy_cos": np.cos(2 * np.pi * dayofyear / 366.0),
        },
        index=timestamps.index,
    )


def _merge_without_duplicate_columns(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Append only columns that are not already present on the left frame."""
    extra_columns = [column for column in right.columns if column not in left.columns]
    if not extra_columns:
        return left.copy()
    return pd.concat([left, right[extra_columns]], axis=1)


@dataclass
class PreprocessorState:
    """Serializable preprocessing state kept inside the checkpoint."""

    context_length: int
    feature_columns: list[str]
    target_mean_by_series: dict[str, float]
    target_std_by_series: dict[str, float]
    feature_mean: dict[str, float]
    feature_std: dict[str, float]
    series_to_idx: dict[str, int]
    global_target_mean: float
    global_target_std: float


class SequencePreprocessor:
    """Build normalized autoregressive features from assignment CSVs."""

    def __init__(self, state: PreprocessorState) -> None:
        self.state = state

    @classmethod
    def fit(cls, train_frame: pd.DataFrame, context_length: int) -> "SequencePreprocessor":
        train = train_frame.copy()
        train[SERIES_COL] = train[SERIES_COL].astype(str)
        numeric_columns = train.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [column for column in numeric_columns if column != TARGET_COL]

        time_features = _base_time_features(train[TIME_COL])
        feature_frame = _merge_without_duplicate_columns(train[feature_columns], time_features)
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

        feature_mean = {
            column: float(feature_frame[column].mean()) if feature_frame[column].notna().any() else 0.0
            for column in feature_frame.columns
        }
        feature_std = {
            column: _safe_std(feature_frame[column].fillna(feature_mean[column]))
            for column in feature_frame.columns
        }

        target_mean_by_series = train.groupby(SERIES_COL)[TARGET_COL].mean().astype(float).to_dict()
        target_std_by_series = train.groupby(SERIES_COL)[TARGET_COL].agg(_safe_std).astype(float).to_dict()
        global_target_mean = float(train[TARGET_COL].mean())
        global_target_std = _safe_std(train[TARGET_COL])
        series_ids = sorted(train[SERIES_COL].unique().tolist())
        state = PreprocessorState(
            context_length=context_length,
            feature_columns=feature_columns,
            target_mean_by_series=target_mean_by_series,
            target_std_by_series=target_std_by_series,
            feature_mean=feature_mean,
            feature_std=feature_std,
            series_to_idx={series_id: idx for idx, series_id in enumerate(series_ids)},
            global_target_mean=global_target_mean,
            global_target_std=global_target_std,
        )
        return cls(state)

    @classmethod
    def from_checkpoint(cls, state_dict: dict[str, Any]) -> "SequencePreprocessor":
        return cls(PreprocessorState(**state_dict))

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "context_length": self.state.context_length,
            "feature_columns": self.state.feature_columns,
            "target_mean_by_series": self.state.target_mean_by_series,
            "target_std_by_series": self.state.target_std_by_series,
            "feature_mean": self.state.feature_mean,
            "feature_std": self.state.feature_std,
            "series_to_idx": self.state.series_to_idx,
            "global_target_mean": self.state.global_target_mean,
            "global_target_std": self.state.global_target_std,
        }

    @property
    def input_size(self) -> int:
        return 1 + len(self.state.feature_mean)

    @property
    def series_count(self) -> int:
        return max(1, len(self.state.series_to_idx))

    def _series_target_stats(self, series_id: str) -> tuple[float, float]:
        mean = self.state.target_mean_by_series.get(series_id, self.state.global_target_mean)
        std = self.state.target_std_by_series.get(series_id, self.state.global_target_std)
        return mean, std

    def _normalize_target(self, series_id: str, values: pd.Series | np.ndarray) -> np.ndarray:
        mean, std = self._series_target_stats(series_id)
        return (np.asarray(values, dtype=np.float32) - mean) / std

    def denormalize_target(self, series_id: str, values: np.ndarray) -> np.ndarray:
        mean, std = self._series_target_stats(series_id)
        return np.asarray(values, dtype=np.float32) * std + mean

    def _prepare_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result[SERIES_COL] = result[SERIES_COL].astype(str)
        for column in self.state.feature_columns:
            if column not in result.columns:
                result[column] = np.nan
        feature_frame = _merge_without_duplicate_columns(
            result[self.state.feature_columns],
            _base_time_features(result[TIME_COL]),
        )
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        for column in feature_frame.columns:
            mean = self.state.feature_mean[column]
            std = self.state.feature_std[column]
            feature_frame[column] = (feature_frame[column].fillna(mean) - mean) / std
        return feature_frame

    def fit_transform_train(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        train = train_frame.copy()
        train[SERIES_COL] = train[SERIES_COL].astype(str)
        train[TIME_COL] = pd.to_datetime(train[TIME_COL])
        train = train.sort_values([SERIES_COL, TIME_COL]).reset_index(drop=True)
        normalized_target = []
        for series_id, part in train.groupby(SERIES_COL, sort=False):
            normalized_target.append(
                pd.Series(self._normalize_target(series_id, part[TARGET_COL]), index=part.index)
            )
        train["_target_norm"] = pd.concat(normalized_target).sort_index()
        feature_frame = self._prepare_feature_frame(train)
        for column in feature_frame.columns:
            train[column] = feature_frame[column].to_numpy()
        return train

    def transform_known_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result[SERIES_COL] = result[SERIES_COL].astype(str)
        result[TIME_COL] = pd.to_datetime(result[TIME_COL])
        feature_frame = self._prepare_feature_frame(result)
        for column in feature_frame.columns:
            result[column] = feature_frame[column].to_numpy()
        return result.sort_values([SERIES_COL, TIME_COL]).reset_index(drop=True)

    def series_index(self, series_id: str) -> int:
        return self.state.series_to_idx.get(series_id, 0)


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """One-step-ahead windows extracted from per-series training data."""

    def __init__(self, frame: pd.DataFrame, preprocessor: SequencePreprocessor, cutoff_by_series: dict[str, pd.Timestamp]):
        self.samples: list[tuple[np.ndarray, int, float]] = []
        feature_columns = ["_target_norm", *preprocessor.state.feature_mean.keys()]
        context = preprocessor.state.context_length

        for series_id, part in frame.groupby(SERIES_COL, sort=False):
            train_part = part.loc[part[TIME_COL] <= cutoff_by_series[series_id]].reset_index(drop=True)
            values = train_part[feature_columns].to_numpy(dtype=np.float32)
            targets = train_part["_target_norm"].to_numpy(dtype=np.float32)
            if len(train_part) <= context:
                continue
            series_index = preprocessor.series_index(series_id)
            for target_idx in range(context, len(train_part)):
                window = values[target_idx - context : target_idx]
                self.samples.append((window, series_index, float(targets[target_idx])))

        if not self.samples:
            raise ValueError("No training samples were created. Reduce context length or provide more history.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, series_index, target = self.samples[idx]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(series_index, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
        )


def split_cutoffs(frame: pd.DataFrame, validation_steps: int) -> dict[str, pd.Timestamp]:
    """Create train/validation cutoff timestamps per series."""
    train_cutoffs: dict[str, pd.Timestamp] = {}
    for series_id, part in frame.groupby(SERIES_COL, sort=False):
        ordered = part.sort_values(TIME_COL).reset_index(drop=True)
        if len(ordered) <= validation_steps:
            train_cutoffs[series_id] = pd.Timestamp(ordered[TIME_COL].iloc[-1])
            continue
        train_cutoffs[series_id] = pd.Timestamp(ordered[TIME_COL].iloc[-validation_steps - 1])
    return train_cutoffs


def evaluate_rollout(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    preprocessor: SequencePreprocessor,
    train_cutoffs: dict[str, pd.Timestamp],
    device: torch.device,
) -> dict[str, float]:
    """Roll forward over the validation tail to estimate sequence quality."""
    feature_columns = list(preprocessor.state.feature_mean.keys())
    maes: list[float] = []
    rmses: list[float] = []

    for series_id, part in frame.groupby(SERIES_COL, sort=False):
        ordered = part.sort_values(TIME_COL).reset_index(drop=True)
        history = ordered.loc[ordered[TIME_COL] <= train_cutoffs[series_id]].copy()
        validation = ordered.loc[ordered[TIME_COL] > train_cutoffs[series_id]].copy()
        if validation.empty or len(history) < preprocessor.state.context_length:
            continue

        history_targets = history[TARGET_COL].to_list()
        predictions: list[float] = []
        for _, row in validation.iterrows():
            feature_history = history[feature_columns].tail(preprocessor.state.context_length).to_numpy(dtype=np.float32)
            target_history = preprocessor._normalize_target(series_id, np.asarray(history_targets[-preprocessor.state.context_length :]))
            window = np.concatenate([target_history[:, None], feature_history], axis=1)
            x_tensor = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
            series_tensor = torch.tensor([preprocessor.series_index(series_id)], dtype=torch.long, device=device)
            with torch.no_grad():
                pred_norm = model(x_tensor, series_tensor).cpu().numpy()
            pred = float(preprocessor.denormalize_target(series_id, pred_norm)[0])
            predictions.append(pred)
            history_targets.append(pred)
            next_row = row.copy()
            next_row[TARGET_COL] = pred
            history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)

        actual = validation[TARGET_COL].to_numpy(dtype=np.float32)
        pred_array = np.asarray(predictions, dtype=np.float32)
        maes.append(float(np.mean(np.abs(pred_array - actual))))
        rmses.append(float(np.sqrt(np.mean((pred_array - actual) ** 2))))

    return {
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "rmse": float(np.mean(rmses)) if rmses else float("nan"),
    }


def load_input_frame(input_dir: Path) -> pd.DataFrame | None:
    """Load future-known covariates if the benchmark provides them."""
    candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def autoregressive_predict(
    model: torch.nn.Module,
    history_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
    preprocessor: SequencePreprocessor,
    future_covariates: pd.DataFrame | None,
    device: torch.device,
) -> pd.DataFrame:
    """Roll the one-step model across the provided forecast index."""
    history = preprocessor.fit_transform_train(history_frame)
    forecast = forecast_index.copy()
    forecast[SERIES_COL] = forecast[SERIES_COL].astype(str)
    forecast[TIME_COL] = pd.to_datetime(forecast[TIME_COL])

    known_future = future_covariates.copy() if future_covariates is not None else forecast.copy()
    known_future[SERIES_COL] = known_future[SERIES_COL].astype(str)
    known_future[TIME_COL] = pd.to_datetime(known_future[TIME_COL])
    if TARGET_COL not in known_future.columns:
        known_future[TARGET_COL] = np.nan
    known_future = known_future.merge(
        forecast[[SERIES_COL, TIME_COL]],
        on=[SERIES_COL, TIME_COL],
        how="right",
    )
    known_future = preprocessor.transform_known_frame(known_future)

    feature_columns = list(preprocessor.state.feature_mean.keys())
    rows: list[pd.DataFrame] = []
    for series_id, index_part in forecast.groupby(SERIES_COL, sort=False):
        history_part = history.loc[history[SERIES_COL] == series_id].copy()
        future_part = known_future.loc[known_future[SERIES_COL] == series_id].copy()
        if len(history_part) < preprocessor.state.context_length:
            raise ValueError(
                f"Series {series_id!r} has only {len(history_part)} history rows; "
                f"need at least context_length={preprocessor.state.context_length}."
            )

        target_history = history_frame.loc[
            history_frame[SERIES_COL].astype(str).eq(series_id)
        ].sort_values(TIME_COL)[TARGET_COL].astype(float).to_list()
        predictions: list[float] = []
        for _, future_row in future_part.iterrows():
            feature_history = history_part[feature_columns].tail(preprocessor.state.context_length).to_numpy(dtype=np.float32)
            norm_target_history = preprocessor._normalize_target(
                series_id,
                np.asarray(target_history[-preprocessor.state.context_length :], dtype=np.float32),
            )
            window = np.concatenate([norm_target_history[:, None], feature_history], axis=1)
            x_tensor = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
            series_tensor = torch.tensor([preprocessor.series_index(series_id)], dtype=torch.long, device=device)
            with torch.no_grad():
                pred_norm = model(x_tensor, series_tensor).cpu().numpy()
            pred = float(preprocessor.denormalize_target(series_id, pred_norm)[0])
            predictions.append(pred)
            target_history.append(pred)

            next_history_row = future_row.copy()
            next_history_row[TARGET_COL] = pred
            history_part = pd.concat([history_part, pd.DataFrame([next_history_row])], ignore_index=True)

        result = index_part[[SERIES_COL, TIME_COL]].copy()
        result["prediction"] = predictions[: len(result)]
        rows.append(result)

    predictions = pd.concat(rows, ignore_index=True)
    predictions[TIME_COL] = predictions[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    return predictions[[SERIES_COL, TIME_COL, "prediction"]]
