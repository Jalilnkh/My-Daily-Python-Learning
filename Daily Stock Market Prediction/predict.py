"""Infer next five trading-session prices and extend the historical price chart.

Put beside train_stock_lightgbm_week.py in src/.
Install: pip install lightgbm pandas numpy scikit-learn matplotlib exchange-calendars
Run: python3 predict.py --ticker NVDA
"""
from pathlib import Path
import argparse
import json

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import exchange_calendars as xcals

# Use exactly the same features as training to avoid inconsistencies.
from train_stock_lightgbm_week import FEATURES, build_features

ROOT = Path(__file__).resolve().parent


def run(ticker, data, models_dir, exchange, recent_days):
    panel = pd.read_csv(data, low_memory=False)
    panel['ticker'] = panel['ticker'].astype(str).str.upper().str.strip()
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce')
    stock = panel.loc[panel['ticker'] == ticker].dropna(subset=['date']).copy()
    if stock.empty:
        raise ValueError(f'{ticker} not found in {data}')
    stock = build_features(stock).reset_index(drop=True)
    latest = stock.iloc[-1]
    last_date = pd.Timestamp(latest['date']).normalize()
    last_close = float(latest['close'])
    if not np.isfinite(last_close) or last_close <= 0:
        raise ValueError('Invalid latest close')

    folder = models_dir / ticker
    meta_file = folder / 'run_info.json'
    if meta_file.is_file():
        metadata = json.loads(meta_file.read_text(encoding='utf-8'))
        if metadata.get('input_features') != FEATURES:
            raise ValueError('Training feature list differs. Retrain models or restore matching trainer.')
        trained_date = pd.Timestamp(metadata['forecast_asof'])
        if last_date < trained_date:
            raise ValueError('Your CSV predates the model training data. Refresh CSV.')
        if last_date > trained_date:
            print('Note: CSV newer than models. Retrain to incorporate the newest labeled days.')

    market = xcals.get_calendar(exchange)
    future = market.sessions_in_range(last_date + pd.Timedelta(days=1),
                                      last_date + pd.Timedelta(days=35))[:5]
    if len(future) != 5:
        raise ValueError('Exchange calendar returned fewer than 5 trading sessions.')
    future = pd.DatetimeIndex(future).tz_localize(None)

    feature_row = stock[FEATURES].iloc[[-1]]
    predictions = []
    for h, future_date in enumerate(future, start=1):
        model_path = folder / f'model_Tplus{h}.txt'
        if not model_path.exists():
            raise FileNotFoundError(f'Missing trained model: {model_path}')
        model = lgb.Booster(model_file=str(model_path))
        if model.feature_name() != FEATURES:
            raise ValueError(f'Feature mismatch in {model_path.name}. Retrain.')
        predicted_return = float(model.predict(feature_row)[0])
        predicted_close = last_close * (1 + predicted_return)
        predictions.append({
            'date': future_date,
            'horizon': h,
            'predicted_return_pct': predicted_return * 100,
            'predicted_close': predicted_close,
        })

    pred = pd.DataFrame(predictions)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / 'inference_next_5_sessions.csv'
    pred.to_csv(csv_path, index=False, date_format='%Y-%m-%d', float_format='%.5f')

    # Prepend last REAL daily close to forecast, so the line joins continuously.
    forecast_x = [last_date] + pred['date'].tolist()
    forecast_y = [last_close] + pred['predicted_close'].tolist()
    for recent, filename in [(False, 'price_history_and_forecast.png'),
                             (True, 'recent_price_and_forecast.png')]:
        part = stock.tail(recent_days) if recent else stock
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(part['date'], part['close'], label='Historical daily close', linewidth=1.6)
        ax.plot(forecast_x, forecast_y, '--o', label='Next 5 trading days', linewidth=2)
        ax.axvline(last_date, linestyle=':', color='gray', label='Last observed close')
        ax.set(title=f'{ticker}: history and 1-week price forecast',
               xlabel='Date', ylabel='Price (listing currency)')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)

    print(f'Last actual close: {last_date.date()} = {last_close:.2f}')
    print(pred.to_string(index=False, formatters={'date': lambda x: x.strftime('%Y-%m-%d')}))
    print(f'Charts and CSV saved to: {folder.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='NVDA')
    parser.add_argument('--data', type=Path, default=ROOT / 'src/output' / 'stock_training_panel.csv')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'src/output' / 'lightgbm_weekly')
    parser.add_argument('--exchange', default='XNYS', help='XNYS for US, XWAR for Poland')
    parser.add_argument('--recent-days', type=int, default=90)
    args = parser.parse_args()
    if args.ticker.upper().endswith('.WA') and args.exchange == 'XNYS':
        args.exchange = 'XWAR'
    run(args.ticker.upper(), args.data, args.models_dir, args.exchange, args.recent_days)