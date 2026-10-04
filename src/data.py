from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SECTOR_GROUPS = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "ORCL", "CSCO"],
    "semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "payments": ["V", "MA"],
    "consumer": ["KO", "PEP", "MCD", "SBUX", "WMT", "COST"],
    "energy": ["XOM", "CVX"],
    "industrials": ["CAT", "DE"],
}


def epoch_seconds(date_string: str) -> int:
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_adjusted_prices(
    tickers: list[str],
    start: str,
    end: str,
    pause_seconds: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download adjusted daily prices from Yahoo's public chart endpoint."""
    series = {}
    errors = {}
    period1 = epoch_seconds(start)
    period2 = epoch_seconds(end)

    for ticker in tickers:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(ticker)}?period1={period1}&period2={period2}"
            "&interval=1d&events=div%2Csplits"
        )
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = payload["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            indicators = result["indicators"]
            adjusted = indicators.get("adjclose", [{}])[0].get("adjclose")
            close = indicators.get("quote", [{}])[0].get("close")
            values = adjusted if adjusted is not None else close
            if not timestamps or values is None:
                raise ValueError("No usable price observations returned")
            index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            series[ticker] = pd.Series(values, index=index, name=ticker)
        except Exception as exc:  # keep the remaining universe usable
            errors[ticker] = str(exc)
        time.sleep(pause_seconds)

    prices = pd.concat(series.values(), axis=1).sort_index() if series else pd.DataFrame()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices, errors


def clean_prices(prices: pd.DataFrame, min_observations: int = 500) -> pd.DataFrame:
    if prices.empty:
        return prices
    usable = prices.notna().sum() >= min_observations
    cleaned = prices.loc[:, usable].copy()
    return cleaned.dropna(how="any")


def unique_group_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for group, tickers in SECTOR_GROUPS.items():
        pairs.extend((a, b, group) for a, b in combinations(tickers, 2))
    return pairs
