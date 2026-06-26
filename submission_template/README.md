# Final Submission Template

This folder now contains a PatchTST baseline that can be trained on the assignment data and packaged as `final_submission.zip`.

## What The Baseline Does

- Trains a one-step autoregressive PatchTST model on `train.csv`.
- Uses the past target history plus timestamp features and any numeric covariates present in the CSV.
- Splits each input channel into patches and encodes them with a shared transformer encoder (channel-independent, as in the PatchTST paper).
- Learns a small embedding per `series_id`.
- Predicts the future horizon recursively so it can fill the full 336-step forecast index.

This is intentionally a simple first research baseline. It is useful for smoke-testing the data pipeline and giving you a PyTorch submission you can iterate on.

## Training

Train a checkpoint from the downloaded dataset:

```bash
python train.py \
  --train /path/to/data/train.csv \
  --checkpoint checkpoint.pt
```

Useful knobs:

- `--context-length 168`: number of historical hourly steps in each input window.
- `--validation-steps 336`: number of tail steps per series reserved for internal validation.
- `--patch-len`, `--stride`: how the context window is split into patches.
- `--d-model`, `--nhead`, `--num-layers`, `--dim-feedforward`, `--dropout`: core transformer capacity settings.
- `--epochs`, `--batch-size`, `--learning-rate`: standard optimization settings.
- `--warmup-steps`, `--min-lr-ratio`: the learning rate ramps up linearly for `--warmup-steps` batches, then cosine-decays toward `--min-lr-ratio * --learning-rate` over the rest of training. The current LR is printed each epoch.
- `--patience`: stop early once validation RMSE hasn't improved for this many epochs (default `5`; set to `<=0` to disable).
- `--resume`: continue training from an existing checkpoint instead of starting from random weights. The model architecture and preprocessing state are loaded from the checkpoint, so other architecture flags (`--patch-len`, `--d-model`, etc.) are ignored when this is set:

  ```bash
  python train.py \
    --train /path/to/data/train.csv \
    --checkpoint checkpoint.pt \
    --resume checkpoint.pt \
    --epochs 10
  ```

During training, the script prints internal validation MAE and RMSE from an autoregressive rollout over the held-out tail of each series.

## Inference Contract

Your submission must support:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

The script:

- reads `forecast_index_test.csv` or `forecast_index_validation.csv`,
- reads `test_input.csv` or `validation_input.csv` when available,
- restores the saved preprocessing state and PatchTST weights,
- writes predictions with schema `series_id,timestamp,prediction`.

## Packaging

From inside this directory, create the archive with:

```bash
zip -r final_submission.zip predict.py requirements.txt checkpoint.pt src
```

Do not include training data, private data, virtual environments, caches, or large unused artifacts.
