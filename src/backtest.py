from __future__ import annotations

import numpy as np
import pandas as pd


def _position_series(z_score: pd.Series, entry: float, exit: float) -> pd.Series:
    position = pd.Series(0.0, index=z_score.index)
    current = 0.0
    for date, z_value in z_score.items():
        if not np.isfinite(z_value):
            position.loc[date] = current
            continue
        if current == 0:
            if z_value < -entry:
                current = 1.0
            elif z_value > entry:
                current = -1.0
        elif current == 1.0 and z_value > -exit:
            current = 0.0
        elif current == -1.0 and z_value < exit:
            current = 0.0
        position.loc[date] = current
    return position


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "maximum_drawdown": np.nan,
            "trading_days": 0,
        }
    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    annualized_return = float(equity.iloc[-1] ** (252 / len(returns)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "trading_days": int(len(returns)),
    }


def extract_trades(returns: pd.Series, held_position: pd.Series) -> pd.DataFrame:
    records = []
    in_trade = False
    start = None
    direction = None
    trade_returns: list[float] = []

    for date in returns.index:
        current_position = float(held_position.loc[date])
        current_return = float(returns.loc[date])

        if not in_trade and current_position != 0:
            in_trade = True
            start = date
            direction = "Long spread" if current_position > 0 else "Short spread"
            trade_returns = [current_return]
        elif in_trade:
            trade_returns.append(current_return)
            if current_position == 0:
                trade_return = float(np.prod(1 + np.array(trade_returns)) - 1)
                records.append({
                    "entry_date": start,
                    "exit_date": date,
                    "direction": direction,
                    "holding_days": len(trade_returns),
                    "return": trade_return,
                })
                in_trade = False
                trade_returns = []

    if in_trade:
        trade_return = float(np.prod(1 + np.array(trade_returns)) - 1)
        records.append({
            "entry_date": start,
            "exit_date": returns.index[-1],
            "direction": direction,
            "holding_days": len(trade_returns),
            "return": trade_return,
        })

    return pd.DataFrame(records)


def run_pair_backtest(
    formation_log_prices: pd.DataFrame,
    test_log_prices: pd.DataFrame,
    pair_row: pd.Series,
    lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    cost_per_dollar: float = 0.0005,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    asset_a = pair_row["asset_a"]
    asset_b = pair_row["asset_b"]
    alpha = float(pair_row["alpha"])
    beta = float(pair_row["beta"])

    combined_log_prices = pd.concat([
        formation_log_prices[[asset_a, asset_b]],
        test_log_prices[[asset_a, asset_b]],
    ]).sort_index()
    spread = combined_log_prices[asset_a] - alpha - beta * combined_log_prices[asset_b]
    rolling_mean = spread.shift(1).rolling(lookback, min_periods=lookback).mean()
    rolling_std = spread.shift(1).rolling(lookback, min_periods=lookback).std()
    z_score = (spread - rolling_mean) / rolling_std
    test_z = z_score.loc[test_log_prices.index].dropna()

    signal_position = _position_series(test_z, entry_threshold, exit_threshold)
    held_position = signal_position.shift(1).fillna(0.0)
    test_returns = test_log_prices.diff()

    beta_abs = abs(beta)
    weight_a = 1 / (1 + beta_abs)
    weight_b = beta_abs / (1 + beta_abs)
    gross_returns = held_position * (
        weight_a * test_returns.loc[held_position.index, asset_a]
        - weight_b * test_returns.loc[held_position.index, asset_b]
    )
    position_change = held_position.diff().fillna(held_position)
    costs = cost_per_dollar * position_change.abs() * (weight_a + weight_b)
    net_returns = (gross_returns - costs).fillna(0.0)
    metrics = performance_metrics(net_returns)
    trades = extract_trades(net_returns, held_position)
    metrics.update({
        "number_of_trades": int(len(trades)),
        "win_rate": float((trades["return"] > 0).mean()) if not trades.empty else np.nan,
        "average_holding_days": float(trades["holding_days"].mean()) if not trades.empty else np.nan,
        "annualized_turnover": float(position_change.abs().sum() / len(net_returns) * 252),
    })
    return net_returns.rename(f"{asset_a}_{asset_b}"), metrics, trades
