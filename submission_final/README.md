# Operations-Load Forecasting — Model Archive

Three trained forecasting models for the `operations_forecasting_2026` task (96 anonymized hourly
operations-unit series, 336-hour validation/test horizon, WAPE-scored). Each model lives in its own
self-contained folder with its own code, its winning checkpoint, its training log, its
`requirements.txt`, and its generated `validation_predictions.csv`.

## Results

| Model | Folder | MAE | RMSE | Notes |
|---|---|---:|---:|---|
| **PatchTST** | [`PatchTST/`](PatchTST/) | **2.493** | **3.458** | Best overall. Channel-independent patch transformer. |
| P_sLSTM | [`P_sLSTM/`](P_sLSTM/) | 3.166 | 4.312 | Patch + sLSTM backbone, channel-independent (as in the original paper). |
| P_sLSTM + channel mixing | [`P_sLSTM_ChannelMixing/`](P_sLSTM_ChannelMixing/) | 3.148 | 4.301 | Same backbone, adds cross-channel information exchange. Best of the two LSTM variants. |

MAE/RMSE are computed on the held-out validation split during training (PatchTST via a genuine
336-step autoregressive rollout on raw target units; P_sLSTM via a single direct 336-step forward
pass, also raw units — the two use different rollout methodologies, so treat cross-family
comparisons as directional, not exact). Full per-epoch curves are in each folder's `logs/`.

### Leaderboard validation scores

Official scores from submitting each model's `validation_predictions.csv` to the Hugging Face
leaderboard (same scorer for all three, so these are directly comparable across models):

| Model | MAE | MSE | RMSE | MAPE | SMAPE | WAPE |
|---|---:|---:|---:|---:|---:|---:|
| **PatchTST** | **2.998666** | **19.291379** | **4.392195** | **28.562516** | **30.701742** | **27.241455** |
| P_sLSTM | 3.559907 | 24.063996 | 4.905507 | 42.328357 | 35.366678 | 32.340062 |
| P_sLSTM + channel mixing | 3.558642 | 24.083054 | 4.907449 | 42.468618 | 35.355530 | 32.328571 |

## What's in each folder

```
PatchTST/
├── train.py, predict.py, src/       # model + training/inference code
├── requirements.txt
├── checkpoints/PatchTST_168_d128_l4_do10.pt
├── logs/LongForecasting/PatchTST_168_d128_l4_do10.log
└── validation_predictions.csv

P_sLSTM/                              # channel-independent (original paper design)
├── run_longExp.py                    # training entrypoint
├── predict_validation.py             # inference entrypoint
├── models/, exp/, data_provider/, layers/, utils/, dataset/
├── requirements.txt
├── checkpoints/P_sLSTM_672_336_wd1e4_do20/checkpoint.pth
├── logs/LongForecasting/P_sLSTM_672_336_wd1e4_do20.log
└── validation_predictions.csv

P_sLSTM_ChannelMixing/                # identical code to P_sLSTM/, channel_mixing=True at train time
└── (same layout, checkpoints/P_sLSTM_672_336_chmix/checkpoint.pth)
```

**On the two P_sLSTM folders:** they contain the *same* implementation. `P_sLSTM_ChannelMixing/` is
not a separate model architecture — it's the identical `models/P_sLSTM.py`, run with
`--channel_mixing True`. Kept as two folders (rather than one, with two checkpoints) so each is
independently self-contained and reproducible without cross-referencing the other. See
["Channel mixing"](#channel-mixing-what-it-actually-does) below for what that flag changes.

A consolidated `requirements.txt` covering all three models also sits at this top level.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or the scoped requirements.txt inside one model folder
```

Each model was trained on a single NVIDIA GPU (CUDA);

## Data

The dataset is **not bundled** in this archive (≈150MB, and sourced externally per the assignment).
Download it from the dataset link in the assignment instructions, then place these files as
`data/train.csv`, `data/forecast_index_validation.csv`, `data/validation_input.csv`,
`data/metadata.json` relative to wherever you run each model's scripts from (i.e. inside
`PatchTST/`, `P_sLSTM/`, or `P_sLSTM_ChannelMixing/` — or adjust the `--train`/path arguments below
to point at wherever you keep it).

## Reproducing PatchTST (best model)

From inside `PatchTST/`:

```bash
python train.py \
  --train path/to/data/train.csv \
  --checkpoint checkpoints/PatchTST_168_d128_l4_do10.pt \
  --context-length 168 \
  --patch-len 24 --stride 12 \
  --d-model 128 --nhead 8 --num-layers 4 \
  --dim-feedforward 256 \
  --embedding-dim 4 \
  --batch-size 128 \
  --dropout 0.1 \
  --patience 10 \
  --epochs 30
```

Generate validation predictions:

```bash
python predict.py \
  --input_dir path/to/data \
  --output_file validation_predictions.csv \
  --checkpoint checkpoints/PatchTST_168_d128_l4_do10.pt
```

## Reproducing P_sLSTM (either variant)

**1. Preprocess the raw data once** (interpolates missing values, label-encodes `series_id`, scales
a subset of columns — from inside `P_sLSTM/` or `P_sLSTM_ChannelMixing/`, with `data/train.csv`
already in place):

```bash
python -c "
import pandas as pd
from dataset.preprocessing import preprocessing
preprocessing(pd.read_csv('data/train.csv'))   # writes ./dataset/train_processed.csv
"
```

**2. Train.** For the channel-independent model (`P_sLSTM/`):

```bash
python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ --data_path train_processed.csv \
  --model_id P_sLSTM_672_336_wd1e4_do20 --model P_sLSTM --data custom \
  --features MS --target target --freq h \
  --seq_len 672 --label_len 24 --pred_len 336 \
  --patch_size 56 --stride 56 \
  --embedding_dim 100 --num_heads 2 --num_blocks 1 \
  --conv1d_kernel_size 8 --group_norm_weight True \
  --channel 23 --enc_in 23 --dec_in 23 --c_out 23 \
  --dropout 0.2 --learning_rate 6e-5 --weight_decay 1e-4 --lradj type1 \
  --batch_size 16 --patience 7 --train_epochs 30
```

For the channel-mixing model (`P_sLSTM_ChannelMixing/`), same command plus one flag:

```bash
  --channel_mixing True \
  --model_id P_sLSTM_672_336_chmix \
```

Checkpoints save automatically to `checkpoints/<model_id>_P_sLSTM_custom_ftMS_sl672_ll24_pl336_0/checkpoint.pth`
whenever validation MSE improves (this framework, unlike PatchTST's, checkpoints incrementally
during training, not just at the end).

**3. Generate validation predictions:**

```bash
python predict_validation.py \
  --checkpoint checkpoints/P_sLSTM_672_336_wd1e4_do20/checkpoint.pth \
  --output validation_predictions.csv \
  --embedding-dim 100 --num-heads 2 --num-blocks 1 --conv1d-kernel-size 8 \
  --seq-len 672 --patch-size 56 --stride 56
```

Add `--channel-mixing` to that command for the `P_sLSTM_ChannelMixing/` checkpoint.

## Channel mixing — what it actually does

`models/P_sLSTM.py` implements the model exactly as published (patch embedding → shared sLSTM
backbone → per-channel projection, fully channel-independent) *unless* `channel_mixing=True`. When
set, one extra stage runs right before the final projection: each channel's patch-encoded
representation is collapsed to a single token, the `C` channel-tokens are treated as a short
sequence, and a second small sLSTM stack strides across that sequence — so channels exchange
information through the same recurrent memory mechanism that normally only mixes information
across time, rather than through a separate attention or MLP layer. This is a lightweight,
from-scratch reproduction of the core idea in *xLSTM-Mixer* (Kraus et al., arXiv:2410.16928); it
does not include that paper's NLinear pre-forecast stage or reversed-order two-view ensembling.
The flag defaults to `False`, so it never affects the original channel-independent model.

Strictly, adding this makes the model no longer "P-sLSTM" as defined in the paper below — channel
independence is one of that paper's two named ingredients. `P_sLSTM_ChannelMixing/` is best
described as a P-sLSTM / xLSTM-Mixer hybrid.

## Packaging the single-model leaderboard submission

If you need the strict single-model `final_submission.zip` for the Hugging Face leaderboard
(`predict.py --input_dir ... --output_file ... --checkpoint ...` contract only — PatchTST was the
best performer, so that's the one to submit):

```bash
cd PatchTST
zip -r ../final_submission.zip predict.py requirements.txt checkpoints/PatchTST_168_d128_l4_do10.pt src
```

Do not include training data, virtual environments, caches, or the other two model folders in that
particular zip — the leaderboard only accepts one model per submission.

## References

- Nie et al., ["A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers"](https://arxiv.org/abs/2211.14730) (PatchTST)
- Kong et al., ["Unlocking the Power of LSTM for Long Term Time Series Forecasting"](https://arxiv.org/abs/2408.10006) (P-sLSTM, AAAI-25)
- Kraus et al., ["xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories"](https://arxiv.org/abs/2410.16928) (channel-mixing idea adapted in `P_sLSTM_ChannelMixing/`)
- Beck et al., [xLSTM](https://github.com/NX-AI/xlstm) (sLSTM backend vendored under `P_sLSTM*/models/xlstm/`)
