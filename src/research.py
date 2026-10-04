from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd


try:
    from statsmodels.tsa.stattools import coint as statsmodels_coint
except Exception:  # optional dependency
    statsmodels_coint = None


@dataclass
class PairTest:
    asset_a: str
    asset_b: str
    group: str
    correlation: float
    alpha: float
    beta: float
    adf_statistic: float
    p_value: float | None
    critical_value_5pct: float
    passes_cointegration: bool

    def to_dict(self) -> dict:
        return asdict(self)


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.replace(0, np.nan)).dropna(how="any")


def log_returns(log_price_frame: pd.DataFrame) -> pd.DataFrame:
    return log_price_frame.diff().dropna(how="any")


def ols_hedge_ratio(asset_a: pd.Series, asset_b: pd.Series) -> tuple[float, float, pd.Series]:
    aligned = pd.concat([asset_a, asset_b], axis=1).dropna()
    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = aligned.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    spread = aligned.iloc[:, 0] - alpha - beta * aligned.iloc[:, 1]
    return float(alpha), float(beta), spread


def _adf_statistic(series: pd.Series, max_lag: int = 1) -> float:
    """Return the ADF t-statistic for a residual with a constant and one lag."""
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 100:
        return np.nan
    delta = np.diff(y)
    lagged = y[:-1]
    rows = [lagged[max_lag:], np.ones(len(lagged) - max_lag)]
    for lag in range(1, max_lag + 1):
        rows.append(delta[max_lag - lag:len(delta) - lag])
    design = np.column_stack(rows)
    target = delta[max_lag:]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residuals = target - design @ coefficients
    dof = max(len(target) - design.shape[1], 1)
    mse = float((residuals @ residuals) / dof)
    covariance = mse * np.linalg.pinv(design.T @ design)
    gamma_se = float(np.sqrt(max(covariance[0, 0], 0)))
    return float(coefficients[0] / gamma_se) if gamma_se else np.nan


def engle_granger_test(
    asset_a: pd.Series,
    asset_b: pd.Series,
    alpha: float,
    beta: float,
    spread: pd.Series,
) -> tuple[float, float | None, float, bool]:
    if statsmodels_coint is not None:
        stat, p_value, critical_values = statsmodels_coint(asset_a, asset_b, trend="c")
        critical_5 = float(critical_values[1])
        return float(stat), float(p_value), critical_5, bool(p_value < 0.05)

    # Approximate Engle-Granger 5% critical value for two series with a constant.
    # This fallback keeps the project runnable without Statsmodels; install the
    # optional requirements for formal MacKinnon p-values.
    stat = _adf_statistic(spread, max_lag=1)
    critical_5 = -3.34
    return stat, None, critical_5, bool(stat < critical_5)


def test_pair(
    asset_a: pd.Series,
    asset_b: pd.Series,
    group: str,
    correlation: float,
) -> PairTest:
    alpha, beta, spread = ols_hedge_ratio(asset_a, asset_b)
    stat, p_value, critical_5, passed = engle_granger_test(
        asset_a, asset_b, alpha, beta, spread
    )
    return PairTest(
        asset_a=asset_a.name,
        asset_b=asset_b.name,
        group=group,
        correlation=float(correlation),
        alpha=alpha,
        beta=beta,
        adf_statistic=stat,
        p_value=p_value,
        critical_value_5pct=critical_5,
        passes_cointegration=passed,
    )


def screen_pairs(
    formation_log_prices: pd.DataFrame,
    grouped_pairs: list[tuple[str, str, str]],
    correlation_threshold: float = 0.70,
    max_pairs: int = 6,
) -> pd.DataFrame:
    returns = log_returns(formation_log_prices)
    tested: list[dict] = []

    for asset_a, asset_b, group in grouped_pairs:
        if asset_a not in formation_log_prices or asset_b not in formation_log_prices:
            continue
        pair_returns = returns[[asset_a, asset_b]].dropna()
        correlation = float(pair_returns[asset_a].corr(pair_returns[asset_b]))
        if not np.isfinite(correlation) or correlation < correlation_threshold:
            continue
        result = test_pair(
            formation_log_prices[asset_a],
            formation_log_prices[asset_b],
            group,
            correlation,
        )
        tested.append(result.to_dict())

    if not tested:
        return pd.DataFrame(columns=list(PairTest.__annotations__.keys()))

    candidates = pd.DataFrame(tested)
    candidates = candidates.sort_values(
        by=["passes_cointegration", "p_value", "adf_statistic", "correlation"],
        ascending=[False, True, True, False],
        na_position="last",
    )

    selected = []
    used_assets: set[str] = set()
    for row in candidates.to_dict("records"):
        if not row["passes_cointegration"]:
            continue
        if row["asset_a"] in used_assets or row["asset_b"] in used_assets:
            continue
        selected.append(row)
        used_assets.update([row["asset_a"], row["asset_b"]])
        if len(selected) >= max_pairs:
            break

    selected_frame = pd.DataFrame(selected)
    if selected_frame.empty:
        return selected_frame
    return selected_frame.reset_index(drop=True)
