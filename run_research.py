from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import run_pair_backtest, performance_metrics  # noqa: E402
from charts import create_charts  # noqa: E402
from data import SECTOR_GROUPS, clean_prices, download_adjusted_prices, unique_group_pairs  # noqa: E402
from research import log_prices, screen_pairs  # noqa: E402


START = "2018-01-01"
END = "2026-01-01"
FORMATION_END = "2022-12-31"
TEST_START = "2023-01-01"


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    tickers = sorted({ticker for group in SECTOR_GROUPS.values() for ticker in group})
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")
    prices, errors = download_adjusted_prices(tickers, START, END)
    if errors:
        print("Download errors:")
        for ticker, error in errors.items():
            print(f"  {ticker}: {error}")
    prices = clean_prices(prices)
    if prices.empty:
        raise RuntimeError("No usable price data was downloaded")

    log_price_frame = log_prices(prices)
    prices.to_csv(results_dir / "adjusted_prices.csv")
    formation = log_price_frame.loc[:FORMATION_END]
    test = log_price_frame.loc[TEST_START:]
    if formation.empty or test.empty:
        raise RuntimeError("Formation or test period is empty")

    selected = screen_pairs(
        formation,
        unique_group_pairs(),
        correlation_threshold=0.70,
        max_pairs=6,
    )
    selected.to_csv(results_dir / "selected_pairs.csv", index=False)

    if selected.empty:
        print("No pairs passed the formation-period screening rules.")
        return

    pair_returns = []
    metric_rows = []
    trade_rows = []
    for _, pair_row in selected.iterrows():
        net_returns, metrics, trades = run_pair_backtest(formation, test, pair_row)
        pair_returns.append(net_returns)
        metric_rows.append({
            "asset_a": pair_row["asset_a"],
            "asset_b": pair_row["asset_b"],
            "group": pair_row["group"],
            **metrics,
        })
        if not trades.empty:
            trades = trades.copy()
            trades.insert(0, "asset_a", pair_row["asset_a"])
            trades.insert(1, "asset_b", pair_row["asset_b"])
            trade_rows.append(trades)

    pair_returns_frame = pd.concat(pair_returns, axis=1).fillna(0.0)
    portfolio_returns = pair_returns_frame.mean(axis=1).rename("portfolio_return")
    portfolio_equity = (1 + portfolio_returns).cumprod().rename("portfolio_equity")
    portfolio_daily = pd.concat([portfolio_returns, portfolio_equity], axis=1)
    portfolio_daily.to_csv(results_dir / "portfolio_daily_returns.csv")

    pair_metrics = pd.DataFrame(metric_rows)
    pair_metrics.to_csv(results_dir / "pair_metrics.csv", index=False)
    portfolio_metrics = pd.DataFrame([performance_metrics(portfolio_returns)])
    portfolio_metrics.to_csv(results_dir / "portfolio_metrics.csv", index=False)

    if trade_rows:
        pd.concat(trade_rows, ignore_index=True).to_csv(results_dir / "pair_trades.csv", index=False)
    else:
        pd.DataFrame().to_csv(results_dir / "pair_trades.csv", index=False)

    create_charts(
        prices=prices,
        selected_pairs=selected,
        pair_metrics=pair_metrics,
        portfolio_daily=portfolio_daily,
        output_dir=results_dir / "charts",
        test_start=TEST_START,
    )

    print("\nSelected pairs:")
    print(selected[["asset_a", "asset_b", "group", "correlation", "adf_statistic", "p_value"]].to_string(index=False))
    print("\nPortfolio metrics:")
    print(portfolio_metrics.to_string(index=False))
    print(f"\nResults written to {results_dir}")


if __name__ == "__main__":
    main()
