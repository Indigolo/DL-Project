"""Generate leaderboard-format validation predictions from the trained P_sLSTM checkpoint.

Reproduces, without persisting them separately, the two scaling stages the
training pipeline applied (dataset/preprocessing.py's StandardScaler over a
subset of columns, then data_provider.Dataset_Custom's per-feature
StandardScaler fit on each series' first 70%), feeds each series' most recent
168-hour window through the model for a single 336-step forward pass, and
inverse-transforms the target channel back to raw units.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.P_sLSTM import Model

TRAIN_CSV = "data/train.csv"
FORECAST_INDEX_CSV = "data/forecast_index_validation.csv"

TARGET = "target"
PRED_LEN = 336

# Must match dataset/preprocessing.py exactly.
PREPROC_SCALED_FEATURES = [
    "trend",
    "workload_intensity",
    "demand_forecast",
    "promotion_intensity",
    "shock_risk",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "nominal_capacity",
    "target",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--conv1d-kernel-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--channel-mixing", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV)
    df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].interpolate().ffill().bfill()

    # Stage 1 scaler (dataset/preprocessing.py).
    scaler1 = StandardScaler()
    df[PREPROC_SCALED_FEATURES] = scaler1.fit_transform(df[PREPROC_SCALED_FEATURES])

    # Stage 2 scaler (data_provider.Dataset_Custom._prepare_panel_series):
    # feature_cols = every column except series_id/date, target moved last.
    feature_cols = [c for c in df.columns if c not in ("series_id", "date")]
    feature_cols = [c for c in feature_cols if c != TARGET] + [TARGET]
    target_channel = feature_cols.index(TARGET)  # last channel

    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    train_slices = []
    for _, group in df.groupby("series_id", sort=False):
        group = group.sort_values("date")
        num_train = int(len(group) * 0.7)
        train_slices.append(group.iloc[:num_train][feature_cols])
    scaler2 = StandardScaler()
    scaler2.fit(pd.concat(train_slices, axis=0).values)

    df[feature_cols] = scaler2.transform(df[feature_cols].values)

    configs = SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=PRED_LEN,
        channel=len(feature_cols),
        embedding_dim=args.embedding_dim,
        patch_size=args.patch_size,
        stride=args.stride,
        num_heads=args.num_heads,
        conv1d_kernel_size=args.conv1d_kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        channel_mixing=args.channel_mixing,
    )
    model = Model(configs)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    forecast_index = pd.read_csv(FORECAST_INDEX_CSV)
    forecast_index["timestamp"] = pd.to_datetime(forecast_index["timestamp"])

    rows = []
    with torch.no_grad():
        for series_id, group in df.groupby("series_id", sort=False):
            group = group.sort_values("date")
            history = group[feature_cols].to_numpy(dtype=np.float32)[-args.seq_len :]
            x = torch.tensor(history[None, :, :], dtype=torch.float32)
            out = model(x).numpy()[0]  # (pred_len, channel)

            pred_target = out[:, target_channel]
            # Undo stage-2 scaling (per-column mean/std over feature_cols).
            pred_target = pred_target * scaler2.scale_[target_channel] + scaler2.mean_[target_channel]
            # Undo stage-1 scaling (per-column mean/std over PREPROC_SCALED_FEATURES).
            idx1 = PREPROC_SCALED_FEATURES.index(TARGET)
            pred_target = pred_target * scaler1.scale_[idx1] + scaler1.mean_[idx1]

            last_date = group["date"].iloc[-1]
            timestamps = pd.date_range(last_date, periods=PRED_LEN + 1, freq="h")[1:]
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "timestamp": timestamps,
                        "prediction": pred_target,
                    }
                )
            )

    predictions = pd.concat(rows, ignore_index=True)

    merged = forecast_index.merge(predictions, on=["series_id", "timestamp"], how="left")
    missing = merged["prediction"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} forecast_index rows had no matching prediction")

    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged.to_csv(args.output, index=False)
    print(f"wrote {args.output} with {len(merged)} rows")


if __name__ == "__main__":
    main()
