#!/usr/bin/env python3
"""Train one daily-price LightGBM model per forecast horizon, per stock.

Input: stock_training_panel.csv written by download_nvda_related_100.py.
At the close of day T, predict adjusted returns for T+1,...,T+5
trading sessions, then convert these returns to illustrative close prices.

Examples:
    python train_stock_lightgbm_week.py --ticker NVDA
    python train.py --ticker AMD
    python train.py --all
    python train.py --data /path/to/stock_training_panel.csv --ticker NVDA

Requires: pip install pandas numpy lightgbm scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "output" / "stock_training_panel.csv"
DEFAULT_OUTDIR = HERE / "output" / "lightgbm_weekly"
HORIZONS = (1, 2, 3, 4, 5)  # Five TRADING sessions, not five calendar days
GAP = max(HORIZONS)          # Purge labels crossing a split boundary
SEED = 42

BASE_FEATURES = [
    "ret_1d",
    "past_ret_5d", "past_ret_10d", "past_ret_20d", "past_ret_60d",
    "past_vol_10d", "past_vol_20d", "past_vol_60d",
    "volume_rel20", "SPY_ret_1d", "QQQ_ret_1d",
]
DERIVED_FEATURES = [
    "intraday_return", "day_range_pct", "overnight_gap_pct",
    "close_location", "excess_ret_vs_spy", "excess_ret_vs_qqq",
    "rsi_14", "day_of_week", "month",
]
FEATURES = BASE_FEATURES + DERIVED_FEATURES


def clean_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def build_features(stock: pd.DataFrame) -> pd.DataFrame:
    """All features use information available at or BEFORE each row's close."""
    df = stock.sort_values("date").drop_duplicates("date", keep="last").copy()
    for col in ["open", "high", "low", "close", "adj_close", "volume"] + BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["intraday_return"] = clean_divide(df["close"], df["open"]) - 1.0
    df["day_range_pct"] = clean_divide(df["high"] - df["low"], df["close"])
    df["overnight_gap_pct"] = clean_divide(df["open"], df["close"].shift(1)) - 1.0
    df["close_location"] = clean_divide(df["close"] - df["low"], df["high"] - df["low"])
    df["excess_ret_vs_spy"] = df["ret_1d"] - df["SPY_ret_1d"]
    df["excess_ret_vs_qqq"] = df["ret_1d"] - df["QQQ_ret_1d"]

    change = df["adj_close"].diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = clean_divide(up, down)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[(down == 0) & (up > 0), "rsi_14"] = 100.0
    df.loc[(down == 0) & (up == 0), "rsi_14"] = 50.0

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # No future-return labels (LABEL_...) can enter FEATURES.
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def new_model(n_estimators: int = 1200) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="l2",
        learning_rate=0.025,
        n_estimators=n_estimators,
        num_leaves=15,
        max_depth=5,
        min_child_samples=35,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=8.0,
        reg_alpha=0.1,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
        importance_type="gain",
    )


def error_report(actual: np.ndarray, predicted: np.ndarray) -> dict:
    baseline = np.zeros_like(actual)  # Predict unchanged adjusted price.
    mae = float(mean_absolute_error(actual, predicted))
    flat_mae = float(mean_absolute_error(actual, baseline))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    directional_accuracy = float(np.mean((predicted > 0) == (actual > 0)))
    return {
        "test_mae_pct_points": round(100 * mae, 4),
        "test_rmse_pct_points": round(100 * rmse, 4),
        "no_change_mae_pct_points": round(100 * flat_mae, 4),
        "mae_improvement_vs_no_change_pct": round(100 * (1 - mae / flat_mae), 2)
            if flat_mae > 0 else None,
        "test_up_down_accuracy_pct": round(100 * directional_accuracy, 2),
    }


def train_one_ticker(panel: pd.DataFrame, ticker: str, root: Path) -> list[dict]:
    stock = panel.loc[panel["ticker"].eq(ticker)].copy()
    if stock.empty:
        print(f"SKIP {ticker}: ticker not in the downloaded CSV")
        return []

    stock = build_features(stock)
    # Keep recent unlabeled observations: the last row is used for forecasting.
    latest = stock.iloc[-1]
    if pd.isna(latest["close"]) or latest["close"] <= 0:
        print(f"SKIP {ticker}: invalid latest closing price")
        return []

    # Use shared known labels for all 5 horizons so every model has the same
    # chronological train/validation/test split.  The last 5 days are
    # necessarily unlabeled for the five-day target.
    label_cols = [f"LABEL_future_return_{h}d" for h in HORIZONS]
    for col in label_cols:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")
    labeled = stock.dropna(subset=label_cols + ["past_vol_60d"]).reset_index(drop=True)

    n = len(labeled)
    if n < 450:
        print(f"SKIP {ticker}: only {n} fully labeled days (minimum 450)")
        return []

    # A strictly chronological split, with 5-row gaps to prevent a label's
    # T+5 outcome from crossing into the next evaluation segment.
    val_begin = int(n * 0.65)
    test_begin = int(n * 0.85)
    train = labeled.iloc[:val_begin - GAP]
    valid = labeled.iloc[val_begin:test_begin - GAP]
    test = labeled.iloc[test_begin:]
    if min(len(train), len(valid), len(test)) < 40:
        print(f"SKIP {ticker}: one time split is too short")
        return []

    folder = root / ticker
    folder.mkdir(parents=True, exist_ok=True)

    forecasts = []
    metrics = []
    history = []
    importances = []
    asof = latest["date"].strftime("%Y-%m-%d")

    for horizon in HORIZONS:
        label = f"LABEL_future_return_{horizon}d"

        evaluation_model = new_model()
        evaluation_model.fit(
            train[FEATURES], train[label],
            eval_set=[(valid[FEATURES], valid[label])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )

        test_pred = evaluation_model.predict(test[FEATURES])
        test_actual = test[label].to_numpy()
        scores = error_report(test_actual, test_pred)
        chosen_trees = max(1, int(evaluation_model.best_iteration_))
        metrics.append({
            "ticker": ticker,
            "horizon_trading_days": horizon,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "train_end": train.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "validation_start": valid.iloc[0]["date"].strftime("%Y-%m-%d"),
            "test_start": test.iloc[0]["date"].strftime("%Y-%m-%d"),
            "best_trees": chosen_trees,
            **scores,
        })

        one_history = pd.DataFrame({
            "ticker": ticker,
            "asof_date": test["date"].dt.strftime("%Y-%m-%d"),
            "horizon_trading_days": horizon,
            "observed_5_day_or_shorter_return": test_actual,
            "predicted_return": test_pred,
            "no_change_return": 0.0,
        })
        history.append(one_history)
        importances.append(pd.DataFrame({
            "horizon": horizon,
            "feature": FEATURES,
            "gain_importance": evaluation_model.feature_importances_,
        }))

        # After holding out the test set for honest performance reporting,
        # retrain a PRODUCTION model on every fully labeled historic day.
        # The test scores above always refer to the evaluation model, not
        # this newly refitted production model.
        production_model = new_model(n_estimators=chosen_trees)
        production_model.fit(labeled[FEATURES], labeled[label])
        production_model.booster_.save_model(str(folder / f"model_Tplus{horizon}.txt"))

        pred_return = float(production_model.predict(stock[FEATURES].iloc[[-1]])[0])
        forecasts.append({
            "ticker": ticker,
            "asof_date": asof,
            "trading_session_ahead": f"T+{horizon}",
            "horizon_trading_days": horizon,
            "last_close": round(float(latest["close"]), 4),
            "predicted_adjusted_return_pct": round(100 * pred_return, 4),
            "illustrative_future_close": round(float(latest["close"]) * (1.0 + pred_return), 4),
        })

    pd.DataFrame(metrics).to_csv(folder / "evaluation.csv", index=False)
    pd.DataFrame(forecasts).to_csv(folder / "forecast_next_5_sessions.csv", index=False)
    pd.concat(history, ignore_index=True).to_csv(folder / "test_predictions.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(folder / "feature_importance.csv", index=False)
    (folder / "run_info.json").write_text(json.dumps({
        "ticker": ticker,
        "forecast_asof": asof,
        "forecast_interpretation": "Predictions made after the last close, in trading-session units",
        "price_note": "Raw close is multiplied by predicted adjusted return; assumes no corporate actions over the next week",
        "training_rows": n,
        "input_features": FEATURES,
        "split": "65% train / 20% validation (approximately) / 15% test, with 5-session purges",
        "evaluation": "Test scores from models fit to train only and tuned on validation only; refit uses all labeled rows afterward",
    }, indent=2), encoding="utf-8")

    print(f"\n{ticker}: forecast made after {asof} closing price {float(latest['close']):.2f}")
    print(pd.DataFrame(forecasts)[[
        "trading_session_ahead", "predicted_adjusted_return_pct", "illustrative_future_close"
    ]].to_string(index=False))
    print("Test MAE (percentage points; lower is better), model versus no-change baseline:")
    print(pd.DataFrame(metrics)[[
        "horizon_trading_days", "test_mae_pct_points", "no_change_mae_pct_points",
        "mae_improvement_vs_no_change_pct", "test_up_down_accuracy_pct"
    ]].to_string(index=False))
    print(f"Saved models, test evaluations and forecasts to: {folder.resolve()}")
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to downloader's stock_training_panel.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--ticker", type=str, default="NVDA",
                        help="One ticker to train (default: NVDA)")
    parser.add_argument("--all", action="store_true",
                        help="Train separate models for each stock in the dataset")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Cannot find {args.data}. Copy this script into the downloader's src/ "
                     "folder, or pass --data /full/path/to/stock_training_panel.csv")

    panel = pd.read_csv(args.data, low_memory=False)
    required = {
        "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
        *BASE_FEATURES,
        *(f"LABEL_future_return_{h}d" for h in HORIZONS),
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        parser.error(f"Missing columns in input CSV: {missing}")

    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()

    tickers = sorted(panel["ticker"].unique()) if args.all else [args.ticker.upper()]
    all_forecasts = []
    for ticker in tickers:
        all_forecasts.extend(train_one_ticker(panel, ticker, args.outdir))

    if all_forecasts:
        args.outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_forecasts).to_csv(args.outdir / "latest_forecasts_all.csv", index=False)
        print(f"\nFinished {len(all_forecasts) // len(HORIZONS)} stocks.")
    else:
        print("No models trained. Check the ticker and daily-history length.")


if __name__ == "__main__":
    main()