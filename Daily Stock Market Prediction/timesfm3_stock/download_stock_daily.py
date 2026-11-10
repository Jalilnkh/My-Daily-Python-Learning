#!/usr/bin/env python3
"""Download daily US/Polish stock prices in the user's existing long CSV schema.

For existing 100-ticker data, skip this and use the existing downloader's
src/output/prices_100_stocks_daily.csv via --data.
"""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import pandas as pd

DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data' / 'prices_100_stocks_daily.csv'


def extract(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        a = frame.columns.get_level_values(0)
        b = frame.columns.get_level_values(1)
        if ticker in a:
            frame = frame.xs(ticker, axis=1, level=0)
        elif ticker in b:
            frame = frame.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    frame.columns = [str(x).strip().lower().replace(' ', '_') for x in frame.columns]
    if 'close' not in frame or frame['close'].dropna().empty:
        return pd.DataFrame()
    if 'adj_close' not in frame.columns:
        frame['adj_close'] = frame['close']
    for name in ['open', 'high', 'low', 'volume']:
        if name not in frame:
            frame[name] = float('nan')
    out = frame[['open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = 'date'
    return out.dropna(subset=['close']).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', default='NVDA,AMD,SPY,QQQ')
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default=None, help='exclusive YYYY-MM-DD; default today (excludes possibly incomplete session)')
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit('Install yfinance: pip install yfinance') from exc
    end = args.end or date.today().isoformat()
    tickers = [x.strip().upper() for x in args.tickers.split(',') if x.strip()]
    if not tickers:
        raise SystemExit('Provide at least one --tickers value')
    results = []
    for t in tickers:
        print(f'Downloading {t} {args.start} <= date < {end}', flush=True)
        raw = yf.download(t, start=args.start, end=end, auto_adjust=False,
                          progress=False, threads=False, timeout=30)
        stock = extract(raw, t)
        if stock.empty:
            print(f'WARNING: {t} has no data, skipping')
            continue
        stock.insert(1, 'ticker', t)
        results.append(stock)
    if not results:
        raise RuntimeError('No stock data downloaded. Check internet/tickers/Yahoo rate limits.')
    panel = pd.concat(results, ignore_index=True).sort_values(['ticker', 'date'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False, float_format='%.8f', date_format='%Y-%m-%d')
    print(f'Saved {len(panel)} daily observations for {panel.ticker.nunique()} tickers: {args.out}')


if __name__ == '__main__':
    main()
