#!/usr/bin/env python3
"""Download a curated 100-ticker US-listed AI-infrastructure universe to CSV.

This DOES NOT label stock declines as caused by the 2026 US election. The basket
contains AI supply-chain stocks, data-center operators, related infrastructure,
and hyperscalers as a comparison group. Empirical co-movement is observational.

Usage:
  pip install yfinance pandas numpy
  python download_nvda_related_100.py --start 2018-01-01 --end 2026-11-07

Writes:
  prices_100_stocks_daily.csv     (OHLCV + adjusted close, one row / ticker / day)
  nvidia_election_event_screen.csv (co-movement over Nov 3-6, 2026)
  stock_training_panel.csv        (past-only features, future-return labels)
  download_status.csv             (missing or unavailable tickers)

`--end` is EXCLUSIVE, so --end 2026-11-07 includes market data on November 6.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
EVENT_PRE_CLOSE = pd.Timestamp('2026-11-03')
EVENT_POST_CLOSE = pd.Timestamp('2026-11-06')
BENCHMARKS = ['SPY', 'QQQ']


def get_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        l0 = raw.columns.get_level_values(0)
        l1 = raw.columns.get_level_values(1)
        if ticker in l0:
            s = raw.xs(ticker, axis=1, level=0)
        elif ticker in l1:
            s = raw.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        s = raw.copy()
    if 'Close' not in s.columns:
        return pd.DataFrame()
    s = s.copy()
    s.columns = [str(c).strip().lower().replace(' ', '_') for c in s.columns]
    for c in ['open','high','low','close','volume']:
        if c not in s.columns:
            s[c] = np.nan
    if 'adj_close' not in s.columns:
        s['adj_close'] = s['close']
    s = s[['open','high','low','close','adj_close','volume']]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.index.name = 'date'
    s = s.dropna(subset=['close']).sort_index()
    return s


def fetch(tickers: list[str], start: str, end: str, batch_size: int = 10):
    all_frames, status = [], []
    for i in range(0,len(tickers),batch_size):
        batch=tickers[i:i+batch_size]
        print(f'Fetching {i+1}-{min(i+batch_size,len(tickers))}/{len(tickers)}: {", ".join(batch)}',flush=True)
        data = pd.DataFrame()
        last_error=''
        for attempt in range(3):
            try:
                data=yf.download(batch,start=start,end=end,auto_adjust=False,
                                 progress=False,threads=False,group_by='ticker',
                                 actions=False,timeout=30)
                if data is not None and not data.empty:
                    break
            except Exception as exc:
                last_error=f'{type(exc).__name__}: {exc}'
            time.sleep(2*(attempt+1))
        for ticker in batch:
            frame=get_ticker_frame(data,ticker)
            if frame.empty:
                # Individual fallback for occasional multi-ticker Yahoo failures
                try:
                    single=yf.download(ticker,start=start,end=end,auto_adjust=False,
                                       progress=False,threads=False,actions=False,timeout=30)
                    frame=get_ticker_frame(single,ticker)
                except Exception as exc:
                    last_error=f'{type(exc).__name__}: {exc}'
            if frame.empty:
                status.append({'ticker':ticker,'status':'NO_PRICE_DATA','rows':0,
                               'note':last_error or 'No daily history returned. Check ticker or listing.'})
                continue
            frame=frame.reset_index()
            frame.insert(1,'ticker',ticker)
            all_frames.append(frame)
            status.append({'ticker':ticker,'status':'OK','rows':len(frame),'note':''})
        time.sleep(0.7)
    if not all_frames:
        raise RuntimeError('No stock prices were downloaded. Check Internet access and Yahoo Finance rate limits.')
    return pd.concat(all_frames,ignore_index=True),pd.DataFrame(status)


def calc_event_screen(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    pv=df.pivot_table(index='date',columns='ticker',values='adj_close',aggfunc='last').sort_index()
    if 'NVDA' not in pv.columns:
        print('WARNING: NVDA missing; cannot build comparison screen.',file=sys.stderr)
        return pd.DataFrame()
    daily=pv.pct_change(fill_method=None)
    pre=daily.loc[:EVENT_PRE_CLOSE].tail(126)  # strictly excludes any reaction after the election
    nvda=pre['NVDA']
    if EVENT_PRE_CLOSE not in pv.index or EVENT_POST_CLOSE not in pv.index:
        print('WARNING: Event close dates missing; event-day columns will contain NA. '
              'Ensure --end >= 2026-11-07 and market data is available.',file=sys.stderr)
    evret={}
    for t in pv.columns:
        try:
            first=pv.at[EVENT_PRE_CLOSE,t]
            last=pv.at[EVENT_POST_CLOSE,t]
            evret[t]=float(last/first-1) if pd.notna(first) and pd.notna(last) and first>0 else np.nan
        except KeyError:
            evret[t]=np.nan
    nvda_ev=evret.get('NVDA',np.nan)
    spy_ev=evret.get('SPY',np.nan)
    qqq_ev=evret.get('QQQ',np.nan)
    rows=[]
    for r in meta.itertuples(index=False):
        t=r.ticker
        if t not in pre.columns:
            continue
        pair=pd.concat([pre[t],nvda],axis=1).dropna()
        if len(pair)>=60 and pair.iloc[:,1].std()>0:
            corr=float(pair.iloc[:,0].corr(pair.iloc[:,1]))
            beta=float(pair.iloc[:,0].cov(pair.iloc[:,1])/pair.iloc[:,1].var())
        else:
            corr,beta=np.nan,np.nan
        er=evret.get(t,np.nan)
        similarly_down=(pd.notna(er) and pd.notna(nvda_ev) and er<0 and nvda_ev<0)
        rows.append({
            'ticker':t,'company':r.company,'group':r.group,
            'pre_election_corr_to_NVDA_126d':round(corr,4) if pd.notna(corr) else np.nan,
            'pre_election_beta_to_NVDA_126d':round(beta,4) if pd.notna(beta) else np.nan,
            'return_Nov3_to_Nov6_pct':round(er*100,3) if pd.notna(er) else np.nan,
            'NVDA_return_Nov3_to_Nov6_pct':round(nvda_ev*100,3) if pd.notna(nvda_ev) else np.nan,
            'SPY_return_Nov3_to_Nov6_pct':round(spy_ev*100,3) if pd.notna(spy_ev) else np.nan,
            'QQQ_return_Nov3_to_Nov6_pct':round(qqq_ev*100,3) if pd.notna(qqq_ev) else np.nan,
            'market_adjusted_event_return_pct':round((er-spy_ev)*100,3) if pd.notna(er) and pd.notna(spy_ev) else np.nan,
            'fell_during_event_like_NVDA':similarly_down,
            'observed_comovement_not_causation':'YES',
            'price_source':'Yahoo Finance via yfinance'
        })
    out=pd.DataFrame(rows)
    # Show similarly negative post-election reactions first, then earlier correlation.
    return out.sort_values(['fell_during_event_like_NVDA','pre_election_corr_to_NVDA_126d'],
                           ascending=[False,False],na_position='last').reset_index(drop=True)


def calc_training_panel(df:pd.DataFrame, meta:pd.DataFrame)->pd.DataFrame:
    # All rolling signals use information available at the end of the current day.
    # Future returns are LABELS (never pass them to the predictor as features).
    stock=df.loc[~df.ticker.isin(BENCHMARKS)].copy()
    stock=stock.sort_values(['ticker','date'])
    group=stock.groupby('ticker',sort=False)
    stock['ret_1d']=group['adj_close'].pct_change(fill_method=None)
    for d in [5,10,20,60]:
        stock[f'past_ret_{d}d']=group['adj_close'].pct_change(d,fill_method=None)
    group=stock.groupby('ticker',sort=False)
    for d in [10,20,60]:
        stock[f'past_vol_{d}d']=group['ret_1d'].transform(lambda x:x.rolling(d,min_periods=d).std())
    stock['volume_rel20']=group['volume'].transform(lambda x:x / x.rolling(20,min_periods=20).mean().replace(0,np.nan))
    # Recompute group for safe future label alignment after sorting.
    group=stock.groupby('ticker',sort=False)
    for horizon in range(1,6):
        future=group['adj_close'].shift(-horizon)
        stock[f'LABEL_future_return_{horizon}d']=future/stock['adj_close']-1
    benchmarks=df.loc[df.ticker.isin(BENCHMARKS),['date','ticker','adj_close']].copy()
    for bench in BENCHMARKS:
        bp=benchmarks.loc[benchmarks.ticker.eq(bench),['date','adj_close']].sort_values('date')
        bp[f'{bench}_ret_1d']=bp['adj_close'].pct_change(fill_method=None)
        bp=bp[['date',f'{bench}_ret_1d']]
        stock=stock.merge(bp,on='date',how='left',validate='many_to_one')
    stock=stock.merge(meta[['ticker','company','group']],on='ticker',how='left',validate='many_to_one')
    # The most recent five days will have incomplete future labels by design.
    return stock


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--universe',type=Path,default=HERE/'ai_infrastructure_100_tickers.csv')
    parser.add_argument('--start',default='2018-01-01')
    parser.add_argument('--end',default='2026-11-07',help='Exclusive end date (YYYY-MM-DD)')
    parser.add_argument('--outdir',type=Path,default=HERE/'output')
    parser.add_argument('--batch-size',type=int,default=10)
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error('batch-size must be positive')
    if pd.Timestamp(args.end)<=pd.Timestamp(args.start):
        parser.error('end must be after start')
    meta=pd.read_csv(args.universe,dtype={'ticker':str})
    required={'ticker','company','group'}
    if not required.issubset(meta.columns):
        parser.error('Universe file must have ticker, company and group columns')
    if meta.ticker.duplicated().any():
        parser.error('Duplicate ticker symbols in universe file')
    if len(meta)!=100:
        print(f'NOTE: universe contains {len(meta)} rather than 100 stocks.',file=sys.stderr)
    tickers=meta.ticker.tolist()
    data,status=fetch(tickers+BENCHMARKS,args.start,args.end,args.batch_size)
    args.outdir.mkdir(parents=True,exist_ok=True)
    status.to_csv(args.outdir/'download_status.csv',index=False)
    prices=data.merge(meta[['ticker','company','group']],on='ticker',how='left')
    prices['date']=prices['date'].dt.strftime('%Y-%m-%d')
    prices.to_csv(args.outdir/'prices_100_stocks_daily.csv',index=False,float_format='%.6f')
    # Convert back to timestamps for time-series operations.
    data['date']=pd.to_datetime(data['date'])
    event=calc_event_screen(data,meta)
    event.to_csv(args.outdir/'nvidia_election_event_screen.csv',index=False)
    training=calc_training_panel(data,meta)
    training['date']=training['date'].dt.strftime('%Y-%m-%d')
    training.to_csv(args.outdir/'stock_training_panel.csv',index=False,float_format='%.8f')
    n_ok=int(status[status.ticker.isin(tickers)].status.eq('OK').sum())
    print(f'Finished: {n_ok}/{len(tickers)} stocks downloaded')
    print(f'OHLCV observations: {len(prices):,}; model panel observations: {len(training):,}')
    print(f'Files saved in: {args.outdir.resolve()}')
    print('IMPORTANT: Co-movement during an election week does not prove the election caused a stock move.')

if __name__=='__main__':
    main()
