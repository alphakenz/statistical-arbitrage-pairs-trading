from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1400
HEIGHT = 800
MARGIN = {"left": 110, "right": 50, "top": 90, "bottom": 90}
COLORS = ["#1f4e79", "#c55a11", "#70ad47", "#7030a0", "#00a6a6"]
GRID = "#d9e1f2"
TEXT = "#1f2937"


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _x_positions(n: int, left: int, right: int) -> np.ndarray:
    if n <= 1:
        return np.array([left], dtype=float)
    return np.linspace(left, WIDTH - right, n)


def _draw_title(draw, title: str):
    draw.text((MARGIN["left"], 28), title, fill=TEXT, font=_font(30))


def _draw_axes(draw, values: np.ndarray):
    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"]
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min, y_max = float(np.nanmin(finite)), float(np.nanmax(finite))
        if y_max == y_min:
            y_min -= 1.0
            y_max += 1.0
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding
    for ratio in np.linspace(0, 1, 6):
        y = bottom - ratio * (bottom - top)
        value = y_min + ratio * (y_max - y_min)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((10, y - 10), f"{value:.2f}", fill=TEXT, font=_font(18))
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    return y_min, y_max, left, right, top, bottom


def _draw_line_chart(
    series: dict[str, pd.Series],
    output: Path,
    title: str,
    vertical_date: str | None = None,
):
    frame = pd.concat(series, axis=1).sort_index()
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    _draw_title(draw, title)
    values = frame.to_numpy(dtype=float)
    y_min, y_max, left, right, top, bottom = _draw_axes(draw, values)
    x = _x_positions(len(frame), left, MARGIN["right"])

    def y_value(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    if vertical_date:
        marker = pd.Timestamp(vertical_date)
        if len(frame.index) and marker >= frame.index.min() and marker <= frame.index.max():
            marker_index = int(frame.index.searchsorted(marker))
            marker_x = x[min(marker_index, len(x) - 1)]
            draw.line((marker_x, top, marker_x, bottom), fill="#7f8c8d", width=2)
            draw.text((marker_x + 6, top + 4), "Out-of-sample", fill=TEXT, font=_font(17))

    for idx, (name, values_series) in enumerate(series.items()):
        values_array = values_series.reindex(frame.index).to_numpy(dtype=float)
        color = COLORS[idx % len(COLORS)]
        previous = None
        for x_value, y_raw in zip(x, values_array):
            if not np.isfinite(y_raw):
                previous = None
                continue
            point = (float(x_value), y_value(float(y_raw)))
            if previous is not None:
                draw.line((*previous, *point), fill=color, width=3)
            previous = point

    legend_x = left
    legend_y = HEIGHT - 48
    for idx, name in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        draw.line((legend_x, legend_y + 9, legend_x + 30, legend_y + 9), fill=color, width=4)
        draw.text((legend_x + 38, legend_y), name, fill=TEXT, font=_font(18))
        legend_x += max(150, len(name) * 11 + 70)

    image.save(output)


def _draw_bar_chart(labels: list[str], values: list[float], output: Path, title: str):
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    _draw_title(draw, title)
    baseline = HEIGHT - 150
    top = 130
    left = 130
    right = WIDTH - 60
    max_abs = max(max(abs(v) for v in values), 0.01)
    scale = (baseline - top) / (2 * max_abs)
    zero_y = baseline - max_abs * scale
    draw.line((left, zero_y, right, zero_y), fill=TEXT, width=2)
    bar_width = (right - left) / max(len(values), 1) * 0.62
    for i, (label, value) in enumerate(zip(labels, values)):
        center = left + (i + 0.5) * (right - left) / len(values)
        height = abs(value) * scale
        y0 = zero_y - height if value >= 0 else zero_y
        y1 = zero_y if value >= 0 else zero_y + height
        color = "#70ad47" if value >= 0 else "#c00000"
        draw.rectangle((center - bar_width / 2, y0, center + bar_width / 2, y1), fill=color)
        draw.text((center - 45, y1 + 12), label, fill=TEXT, font=_font(18))
        draw.text((center - 45, y0 - 28), f"{value:.1%}", fill=TEXT, font=_font(17))
    image.save(output)


def create_charts(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    output_dir: Path,
    test_start: str = "2023-01-01",
    lookback: int = 60,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if prices.empty:
        return

    for _, pair in selected_pairs.iterrows():
        asset_a = pair["asset_a"]
        asset_b = pair["asset_b"]
        if asset_a not in prices or asset_b not in prices:
            continue
        pair_prices = prices[[asset_a, asset_b]].dropna()
        normalized = pair_prices / pair_prices.iloc[0] * 100
        safe_name = f"{asset_a}_{asset_b}"
        _draw_line_chart(
            {asset_a: normalized[asset_a], asset_b: normalized[asset_b]},
            output_dir / f"{safe_name}_normalized_prices.png",
            f"{asset_a} / {asset_b} Normalized Adjusted Prices",
            vertical_date=test_start,
        )

        log_frame = np.log(pair_prices)
        spread = log_frame[asset_a] - float(pair["alpha"]) - float(pair["beta"]) * log_frame[asset_b]
        rolling_mean = spread.shift(1).rolling(lookback, min_periods=lookback).mean()
        rolling_std = spread.shift(1).rolling(lookback, min_periods=lookback).std()
        z_score = ((spread - rolling_mean) / rolling_std).loc[test_start:]
        z_frame = pd.DataFrame({"Z-score": z_score})
        z_frame["Entry +2"] = 2.0
        z_frame["Entry -2"] = -2.0
        z_frame["Exit +0.5"] = 0.5
        z_frame["Exit -0.5"] = -0.5
        _draw_line_chart(
            {name: z_frame[name] for name in z_frame.columns},
            output_dir / f"{safe_name}_zscore_signals.png",
            f"{asset_a} / {asset_b} Out-of-Sample Z-Score",
        )

    if not portfolio_daily.empty:
        equity = portfolio_daily["portfolio_equity"]
        drawdown = equity / equity.cummax() - 1
        _draw_line_chart(
            {"Portfolio equity": equity},
            output_dir / "portfolio_equity.png",
            "Equal-Weighted Portfolio Equity Curve",
        )
        _draw_line_chart(
            {"Drawdown": drawdown},
            output_dir / "portfolio_drawdown.png",
            "Portfolio Drawdown",
        )

    if not pair_metrics.empty:
        labels = [f"{row.asset_a}/{row.asset_b}" for row in pair_metrics.itertuples()]
        values = pair_metrics["annualized_return"].astype(float).tolist()
        _draw_bar_chart(
            labels,
            values,
            output_dir / "pair_annualized_returns.png",
            "Out-of-Sample Annualized Return by Pair",
        )
