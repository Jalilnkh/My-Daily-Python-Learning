#!/usr/bin/env python3
"""Train/evaluate TimesFM-3 prediction head on daily stock prices, forecast 5 sessions.

Official TimesFM3Torch.forward() is differentiable; .decode() is @torch.no_grad().
This script therefore uses the same context-only forward path for BOTH training and
inference, extracting the first H future-price forecasts from the last context patch.

Requires local installation of official timesfm>=3.0 and downloaded model checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Keep optional heavyweight imports out of module import so data tests can run alone.
DEFAULT_INPUT = Path(__file__).resolve().parent / 'data' / 'prices_100_stocks_daily.csv'
DEFAULT_RESULTS = Path(__file__).resolve().parent / 'results'
MODEL_ID = 'google/timesfm-3.0-pytorch'


def read_series(csv_path: Path, ticker: str, column: str = 'adj_close') -> pd.DataFrame:
    """Load stock's daily series from user's downloader or our standalone downloader."""
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing {csv_path}. Download data or supply --data FILE')
    raw = pd.read_csv(csv_path, low_memory=False)
    required = {'date', 'ticker', column}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing columns: {sorted(missing)}')
    stock = raw.loc[raw['ticker'].astype(str).str.upper().eq(ticker.upper()),
                    ['date', column]].copy()
    stock['date'] = pd.to_datetime(stock['date'], errors='coerce')
    stock['price'] = pd.to_numeric(stock[column], errors='coerce')
    stock = (stock.dropna(subset=['date', 'price'])
                  .loc[lambda df: (df['price'] > 0) & np.isfinite(df['price'])]
                  .sort_values('date').drop_duplicates('date', keep='last')
                  [['date', 'price']].reset_index(drop=True))
    if len(stock) < 350:
        raise ValueError(f'{ticker}: only {len(stock)} valid daily prices; need >=350')
    return stock


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    val_start: int
    test_start: int


def split_origin_indices(n: int, context: int, horizon: int,
                         val_days: int, test_days: int, train_stride: int) -> Splits:
    """No target crosses partitions. Validation/test contain only later targets.

    Origin i has context [i-context+1 ... i], target [i+1 ... i+horizon].
    A train target must end BEFORE the validation boundary. A validation target
    must start at/after validation boundary and end BEFORE test boundary.
    """
    if context < 32 or context % 32:
        raise ValueError('--context must be >=32 and a multiple of 32 (TimesFM-3 patch size)')
    if not 1 <= horizon <= 64:
        raise ValueError('--horizon must be between 1 and 64')
    if min(val_days, test_days) < horizon or train_stride < 1:
        raise ValueError('val-days and test-days must be >= horizon; stride >= 1')
    val_start = n - val_days - test_days
    test_start = n - test_days
    idx = np.arange(context - 1, n - horizon)
    train = idx[idx + horizon < val_start][::train_stride]
    val = idx[(idx + 1 >= val_start) & (idx + horizon < test_start)]
    test = idx[(idx + 1 >= test_start) & (idx + horizon < n)]
    if min(len(train), len(val), len(test)) < 5:
        raise ValueError(f'Insufficient train/val/test windows: {len(train)}, '
                         f'{len(val)}, {len(test)}. Lower context/split sizes.')
    return Splits(train, val, test, val_start, test_start)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(prices: np.ndarray, origins: np.ndarray,
                context: int, horizon: int, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class DailyWindowDataset(Dataset):
        def __len__(self):
            return len(origins)

        def __getitem__(self, j):
            i = int(origins[j])
            x = prices[i - context + 1:i + 1].copy()
            y = prices[i + 1:i + horizon + 1].copy()
            return torch.from_numpy(x), torch.from_numpy(y), i

    return DataLoader(DailyWindowDataset(), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def forward_prices(model, x, horizon):
    """Differentiable prediction from official TimesFM-3 full-sequence forward.

    x: (batch, context); context divisible by 32. No label/future data is passed.
    One input variate (target), all context values observed. output logits have
    shape (batch, variates, patches, 64, 9 quantiles).

    For horizon<=32, official decode starts with the last context patch and
    extracts its first `horizon` output positions. We match this extraction,
    leaving out decode() since its decorator disables gradients.
    """
    import torch
    if x.ndim != 2 or x.shape[1] % model.input_patch_len:
        raise ValueError('Input must be [batch, context] with whole 32-day patches')
    batch, context = x.shape
    n_patches = context // model.input_patch_len
    vals = x.reshape(batch, 1, n_patches, model.input_patch_len)
    masks = torch.zeros_like(vals, dtype=torch.bool)
    targets = torch.ones((batch, 1, n_patches), device=x.device, dtype=torch.bool)
    logits = model({'values': vals, 'masks': masks, 'patch_is_target': targets})['logits']
    if horizon > model.input_patch_len:
        raise ValueError('This context-only implementation supports horizon<=32')
    return logits[:, 0, -1, :horizon, model.num_quantiles // 2]


def relative_huber(pred, actual, base):
    import torch.nn.functional as F
    return F.smooth_l1_loss(pred / base - 1.0, actual / base - 1.0, beta=0.01)


def loop(model, loader, device: str, horizon: int, optimizer=None):
    import torch
    is_training = optimizer is not None
    # eval() is intentional: deterministic frozen backbone; it DOES NOT prevent
    # autograd on output_head. The context-only forward has no no_grad decorator.
    model.eval()
    sq, abs_sum, baseline_abs, counts, losses = 0., 0., 0., 0, 0.
    for x, y, _ in loader:
        x, y = x.to(device=device), y.to(device=device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            pred = forward_prices(model, x, horizon)
            loss = relative_huber(pred, y, x[:, -1:].contiguous())
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.output_head.parameters(), 1.0)
                optimizer.step()
        diff = (pred.detach() - y).float()
        base_diff = (x[:, -1:] - y).float()
        sq += float(diff.square().sum().item())
        abs_sum += float(diff.abs().sum().item())
        baseline_abs += float(base_diff.abs().sum().item())
        n = y.numel()
        counts += n
        losses += float(loss.detach().item()) * n
    return {'loss': losses / counts, 'mae': abs_sum / counts,
            'rmse': float(np.sqrt(sq / counts)), 'naive_mae': baseline_abs / counts}


def evaluate_rows(model, loader, stock: pd.DataFrame, horizon: int, device: str,
                  max_batches: int | None = None) -> pd.DataFrame:
    import torch
    rows = []
    with torch.no_grad():
        for b, (x, y, origins) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            preds = forward_prices(model, x.to(device), horizon).cpu().numpy()
            for j, i in enumerate(origins.numpy()):
                i = int(i)
                for step in range(horizon):
                    rows.append({
                        'asof_date': stock.loc[i, 'date'].strftime('%Y-%m-%d'),
                        'target_date': stock.loc[i + step + 1, 'date'].strftime('%Y-%m-%d'),
                        'horizon': step + 1,
                        'last_price': float(x[j, -1].item()),
                        'actual_price': float(y[j, step].item()),
                        'pred_price': float(preds[j, step]),
                        'naive_price': float(x[j, -1].item())
                    })
    return pd.DataFrame(rows)


def save_training_curves(frame: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame['epoch'], frame['train_mae'], label='Train MAE')
    ax.plot(frame['epoch'], frame['val_mae'], label='Validation MAE')
    ax.axhline(float(frame['val_naive_mae'].iloc[0]), linestyle=':',
               label='Validation unchanged-price baseline')
    ax.set(xlabel='Epoch', ylabel='MAE (price units)', title='TimesFM-3 fine-tuning')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def future_sessions(last_date: pd.Timestamp, count: int, exchange: str) -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    cal = xcals.get_calendar(exchange)
    # Future sessions may pass the installed calendar's upper bound. Raise a
    # specific error instead of silently predicting holidays as sessions.
    start = (pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1))
    end = start + pd.Timedelta(days=70)
    dates = cal.sessions_in_range(start, end).tz_localize(None)
    if len(dates) < count:
        raise RuntimeError(f'Calendar {exchange} has fewer than {count} future sessions')
    return dates[:count]


def produce_forecast(model, stock: pd.DataFrame, context: int, horizon: int,
                     device: str, out: Path, ticker: str, exchange: str,
                     plot_days: int, price_col: str):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(stock) < context:
        raise ValueError('Not enough rows for context')
    x = torch.from_numpy(stock['price'].tail(context).to_numpy(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        yhat = forward_prices(model, x, horizon).cpu().numpy().reshape(-1)
    dates = future_sessions(stock['date'].iloc[-1], horizon, exchange)
    last_date = stock['date'].iloc[-1]
    last_price = float(stock['price'].iloc[-1])
    df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'ticker': ticker,
                       'horizon': np.arange(1, horizon + 1),
                       'predicted_price': yhat,
                       'return_from_last_close_pct': 100 * (yhat / last_price - 1)})
    df.to_csv(out / 'forecast_next_week.csv', index=False)
    hist = stock.tail(plot_days)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist['date'], hist['price'], label=f'Observed {price_col}', linewidth=1.6)
    ax.plot([last_date] + list(dates), [last_price] + list(yhat), '--o',
            label=f'TimesFM-3 fine-tuned (next {horizon} sessions)', linewidth=2)
    ax.axvline(last_date, linestyle=':', alpha=.7)
    ax.set(xlabel='Trading date', ylabel='Price (listing currency)',
           title=f'{ticker}: historical daily price and next-week forecast')
    ax.grid(alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / 'history_plus_forecast.png', dpi=160)
    plt.close(fig)
    print(f'Last observed {price_col}: {last_date.date()} {last_price:.3f}')
    print(df.to_string(index=False))
    print(f'Forecast saved in {out.resolve()}')
    return df


def load_official_model(device: str):
    try:
        from timesfm3.torch.model import TimesFM3Torch
    except ImportError as exc:
        raise RuntimeError('Install official TimesFM-3: pip install "timesfm[torch]>=3.0.0"') from exc
    print(f'Loading official {MODEL_ID}; initial run downloads pretrained weights.')
    model = TimesFM3Torch.from_pretrained(MODEL_ID)
    if not hasattr(model, 'output_head'):
        raise RuntimeError('Official TimesFM-3 structure changed (missing output_head).')
    model = model.to(device)
    model.eval()
    return model


def train(args):
    import torch
    seed_everything(args.seed)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        warnings.warn('TimesFM-3 (~330M parameters) is slow on CPU; a CUDA GPU is recommended.')
    stock = read_series(args.data, args.ticker, args.price_col)
    splits = split_origin_indices(len(stock), args.context, args.horizon,
                                  args.val_days, args.test_days, args.stride)
    p = stock['price'].to_numpy(np.float32)
    out = args.outdir / args.ticker.upper()
    out.mkdir(parents=True, exist_ok=True)
    loaders = {
        'train': make_loader(p, splits.train, args.context, args.horizon, args.batch_size, True),
        'val': make_loader(p, splits.val, args.context, args.horizon, args.batch_size, False),
        'test': make_loader(p, splits.test, args.context, args.horizon, args.batch_size, False),
    }
    print(f'Windows: train {len(splits.train)}, val {len(splits.val)}, test {len(splits.test)}')
    print(f'Validation targets begin {stock.date.iloc[splits.val_start].date()}, '
          f'test targets begin {stock.date.iloc[splits.test_start].date()}')
    model = load_official_model(device)
    if args.context % model.input_patch_len:
        raise ValueError(f'Context must be a multiple of model patch {model.input_patch_len}')
    if args.horizon > model.input_patch_len:
        raise ValueError(f'Horizon must be <= model input patch size ({model.input_patch_len})')
    # Actual fine-tuning of TimesFM-3's own prediction head. No training of a
    # separate external model; the pretrained transformer body is frozen.
    model.requires_grad_(False)
    model.output_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.output_head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    print('Trainable TimesFM-3 parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Fixed zero-shot comparison, before changing any weights.
    zero_shot_test = loop(model, loaders['test'], device, args.horizon)
    print('Zero-shot TEST MAE (before training):', round(zero_shot_test['mae'], 5))
    best = float('inf')
    stale = 0
    history = []
    best_weights = None
    for epoch in range(1, args.epochs + 1):
        tr = loop(model, loaders['train'], device, args.horizon, optimizer)
        with torch.no_grad():
            va = loop(model, loaders['val'], device, args.horizon)
        history.append({'epoch': epoch, 'train_loss': tr['loss'], 'train_mae': tr['mae'],
                        'val_loss': va['loss'], 'val_mae': va['mae'],
                        'val_naive_mae': va['naive_mae']})
        print(f"Epoch {epoch:02d}/{args.epochs:02d}  train MAE={tr['mae']:.4f}  "
              f"val MAE={va['mae']:.4f}  val naive={va['naive_mae']:.4f}", flush=True)
        if va['mae'] < best - args.min_delta:
            best = va['mae']
            stale = 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.output_head.state_dict().items()}
            torch.save(best_weights, out / 'best_timesfm3_head.pt')
        else:
            stale += 1
        if stale >= args.patience:
            print(f'Early stopping after {epoch} epochs (no validation improvement).')
            break
    if best_weights is None:
        raise RuntimeError('No valid model was saved')
    model.output_head.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test = loop(model, loaders['test'], device, args.horizon)
    details = evaluate_rows(model, loaders['test'], stock, args.horizon, device)
    details.to_csv(out / 'heldout_test_predictions.csv', index=False)
    frame = pd.DataFrame(history)
    frame.to_csv(out / 'training_history.csv', index=False)
    save_training_curves(frame, out / 'training_curve.png')
    summary = {
        'model': MODEL_ID, 'fine_tuned': 'official TimesFM3Torch.output_head',
        'ticker': args.ticker.upper(), 'price_col': args.price_col,
        'context': args.context, 'horizon': args.horizon,
        'epochs_requested': args.epochs, 'epochs_completed': len(history),
        'train_windows': len(splits.train), 'val_windows': len(splits.val),
        'test_windows': len(splits.test), 'val_start': stock.date.iloc[splits.val_start].strftime('%Y-%m-%d'),
        'test_start': stock.date.iloc[splits.test_start].strftime('%Y-%m-%d'),
        'last_available_date': stock.date.iloc[-1].strftime('%Y-%m-%d'),
        'train_stride': args.stride, 'zero_shot_test_mae': zero_shot_test['mae'],
        'fine_tuned_test_mae': test['mae'], 'fine_tuned_test_rmse': test['rmse'],
        'unchanged_price_test_mae': test['naive_mae'],
        'note': 'Retrospective held-out metrics. Do not assume profitable real-world forecasts. '
                'Test set was not used to select epochs; zero-shot comparison uses it for reporting only.'
    }
    (out / 'run_info.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'zero_shot_test_mae': zero_shot_test['mae'],
                      'fine_tuned_test_mae': test['mae'],
                      'naive_test_mae': test['naive_mae']}, indent=2))
    if test['mae'] >= test['naive_mae']:
        print('WARNING: Fine-tuned model underperforms unchanged-price baseline on test. '
              'Do not use as a trading signal without further evaluation.')
    produce_forecast(model, stock, args.context, args.horizon, device, out,
                     args.ticker.upper(), args.exchange, args.plot_days, args.price_col)


def predict(args):
    import torch
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.outdir / args.ticker.upper()
    meta_path = out / 'run_info.json'
    head_path = out / 'best_timesfm3_head.pt'
    if not head_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing trained checkpoint in {out}. Run train first.')
    meta = json.loads(meta_path.read_text())
    if meta['ticker'] != args.ticker.upper():
        raise ValueError('Trained model ticker differs')
    stock = read_series(args.data, args.ticker, meta['price_col'])
    model = load_official_model(device)
    state = torch.load(head_path, map_location='cpu', weights_only=True)
    model.output_head.load_state_dict(state, strict=True)
    model.eval()
    if stock['date'].iloc[-1].strftime('%Y-%m-%d') > meta['last_available_date']:
        print('NOTE: newer prices available; inference uses them, but model weights were not retrained.')
    produce_forecast(model, stock, int(meta['context']), int(meta['horizon']),
                     device, out, args.ticker.upper(), args.exchange, args.plot_days,
                     meta['price_col'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--data', type=Path, default=DEFAULT_INPUT,
                        help='CSV generated by our downloader OR download_nvda_related_100.py')
    parser.add_argument('--ticker', default='NVDA')
    parser.add_argument('--price-col', default='adj_close', choices=['adj_close', 'close'])
    parser.add_argument('--outdir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-5)
    parser.add_argument('--val-days', type=int, default=126)
    parser.add_argument('--test-days', type=int, default=126)
    parser.add_argument('--stride', type=int, default=5, help='Training window origin spacing')
    parser.add_argument('--exchange', default='XNYS', choices=['XNYS', 'XWAR'])
    parser.add_argument('--plot-days', type=int, default=180)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.ticker.upper().endswith('.WA') and args.exchange == 'XNYS':
        args.exchange = 'XWAR'
    if args.command == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
