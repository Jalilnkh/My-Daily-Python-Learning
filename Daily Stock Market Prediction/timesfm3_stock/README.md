# TimesFM-3 Stock Forecasting — Real PyTorch Fine-Tuning

This project uses Google's **TimesFM-3.0** pretrained **PyTorch** checkpoint, trains
its own **`output_head` parameters with AdamW, backpropagation, epochs and early stopping**,
then predicts the next **five actual stock-exchange sessions** and saves a historical
price + predicted future line plot.

> **What it is / isn't:** This is **head-only fine-tuning**, not pretraining the
> 330-million-parameter model from scratch. The Transformer backbone is frozen.
> The official `TimesFM3Torch.decode()` is wrapped in `@torch.no_grad()`;
> therefore `timesfm3_stock.py` calls the differentiable `TimesFM3Torch.forward()`
> directly during training **and** inference. For 5 steps it extracts predictions
> from the last 32-day context patch, matching the first forecast patch of the
> official decode path. It does not use `TimesFM3Evaluator` (inference-only).
> The forward-only path does **not** perform decode's optional linear detrending,
> so its results need not match the stock TimesFM-3 evaluator exactly.

## Requirements

- Python 3.10 or newer, internet **on your computer** for Yahoo Finance and
  model checkpoint downloads, ideally an NVIDIA GPU. First checkpoint load
  downloads the **~330M parameter pretrained model** (substantial disk/RAM).
- Self-hosted TimesFM-3 weights: **non-commercial, non-production license**.
  Google Cloud has its own terms for commercial use. Review the model license.
- Stock predictions are experimental and may be no better than an unchanged-price baseline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Option A — Download NVDA/AMD/SPY/QQQ automatically

From this package directory:

```bash
python download_stock_daily.py --tickers NVDA,AMD,SPY,QQQ --start 2015-01-01
python timesfm3_stock.py train --ticker NVDA --epochs 20 --context 256 --horizon 5 --batch-size 8
```

The downloader fetches adjusted OHLCV; output is `data/prices_100_stocks_daily.csv`.
Re-running with the default end date downloads sessions strictly before today
(end date is exclusive), avoiding incomplete daily candles during market hours.
To include today's finalized close, use `--end` tomorrow **after** the market closes.

## Option B — Use your existing GitHub downloader

Your previous script `download_nvda_related_100.py` already generates
`src/output/prices_100_stocks_daily.csv` (and `stock_training_panel.csv`).
Use **prices** — the raw daily date/ticker/adj_close series — not the feature/label
panel, because TimesFM-3 receives sequential historical values directly.

From `Daily Stock Market Prediction`:

```bash
python src/download_nvda_related_100.py --start 2015-01-01 --end 2026-11-10
python timesfm3_stock.py train --data src/output/prices_100_stocks_daily.csv --ticker NVDA --epochs 20
```

This example excludes the possibly incomplete November 10 US trading session.
Place `timesfm3_stock.py` in the project root for the command above, or adjust
its path if you place it in `src/`. Your original downloader requires the ticker
universe file at `dataset/to_download/ai_infrastructure_100_tickers.csv`.

## Independent inference (after training)

```bash
python timesfm3_stock.py predict --ticker NVDA
```

Or using your existing CSV:

```bash
python timesfm3_stock.py predict --ticker NVDA --data src/output/prices_100_stocks_daily.csv
```

## Train and evaluate Polish ORLEN

```bash
python download_stock_daily.py --tickers PKN.WA --start 2015-01-01
python timesfm3_stock.py train --ticker PKN.WA --exchange XWAR
```

**Note:** Verify ORLEN's current Yahoo Finance symbol before running.

## Time series design / no leakage

- Lookback = 256 daily observations (one trading day per timestep).
- Labels = **next five** close prices.
- Each origin `t`: model only receives `[t-255, ..., t]`.
- We hold out the **latest 126 observations** for test and prior **126** for validation.
- Train windows must have *all five targets* before validation begins.
- Validation windows must have all five targets before test begins.
- Training origins use a configurable `--stride 5`; validation and test are daily.
- Early stopping selects the best checkpoint **only on validation**.
- MAE versus no-change (`predicted_price = last_observed_price`) and zero-shot
  pretrained TimesFM-3 are reported using a historical holdout.
- The final forecast uses the latest daily price as available *at inference*.

## Training hyperparameters

```bash
python timesfm3_stock.py train \
  --ticker NVDA \
  --context 256 \
  --horizon 5 \
  --epochs 20 \
  --batch-size 8 \
  --stride 5 \
  --lr 0.00003 \
  --patience 5 \
  --val-days 126 \
  --test-days 126
```

If compute is slow, try `--batch-size 2 --stride 10 --epochs 8`. The full
backbone is still executed each batch, so CPU fine-tuning can be very slow.

## Generated files

```
results/NVDA/
  best_timesfm3_head.pt       # Actual trained TimesFM3 output head weights
  run_info.json               # hyperparameters, split dates, test metrics
  training_history.csv        # epoch-wise loss and MAE
  training_curve.png          # training and validation curves
  heldout_test_predictions.csv# historical OOS predictions
  forecast_next_week.csv      # 5 next exchange-trading sessions
  history_plus_forecast.png   # price-history continuation chart
```

Data files are **not** shipped with the code (download on your machine).
The model weights are **not** shipped; the official checkpoint downloads when
first run. Since this environment has no internet to download the checkpoint
or the latest Yahoo data, the full TimesFM3 training run cannot be executed
here. Data-splitting and script smoke tests use an artificial toy model and
synthetic data only, without presenting them as stock forecasts.

Official sources:
- https://github.com/google-research/timesfm
- https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/model.py
- https://huggingface.co/google/timesfm-3.0-pytorch
